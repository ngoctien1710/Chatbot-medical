const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = Number(process.env.PORT || 3000);
const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL || 'http://127.0.0.1:8001';
const RAG_SERVICE_TIMEOUT_MS = Number(process.env.RAG_SERVICE_TIMEOUT_MS || 120000);
const CHAT_LOG_ROOT = path.join(__dirname, '..', 'chat_log');

// Middleware
app.use(cors());
app.use(express.json());

const SESSION_ID_REGEX = /^(\d{2}-\d{2}-\d{4})-session-(\d{3})$/;

const pad3 = (value) => String(value).padStart(3, '0');

const ensureDir = (dirPath) => {
    fs.mkdirSync(dirPath, { recursive: true });
};

const todayDateKey = () => {
    const now = new Date();
    const day = String(now.getDate()).padStart(2, '0');
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const year = String(now.getFullYear());
    return `${day}-${month}-${year}`;
};

const parseDateKey = (dateKey) => {
    const match = /^(\d{2})-(\d{2})-(\d{4})$/.exec(dateKey);
    if (!match) {
        return null;
    }
    return {
        day: match[1],
        month: match[2],
        year: match[3],
    };
};

const dateFolderPath = (dateKey) => {
    const parsed = parseDateKey(dateKey);
    if (!parsed) {
        throw new Error(`Invalid date format: ${dateKey}`);
    }
    return path.join(CHAT_LOG_ROOT, parsed.day + '-' + parsed.month + '-' + parsed.year);
};

const buildSessionFileName = (sequence) => `session_${pad3(sequence)}.json`;

const buildSessionId = (dateKey, sequence) => `${dateKey}-session-${pad3(sequence)}`;

const parseSessionId = (sessionId) => {
    const match = SESSION_ID_REGEX.exec(sessionId);
    if (!match) {
        return null;
    }
    return {
        dateKey: match[1],
        sequence: Number(match[2]),
    };
};

const readJsonFile = (filePath) => {
    const raw = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(raw);
};

const writeJsonFile = (filePath, payload, options = {}) => {
    fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), options);
};

const sessionFilesForDate = (dateKey) => {
    const datePath = dateFolderPath(dateKey);
    if (!fs.existsSync(datePath)) {
        return [];
    }

    return fs
        .readdirSync(datePath)
        .filter((fileName) => /^session_\d{3}\.json$/.test(fileName))
        .sort((a, b) => {
            const aNum = Number(a.match(/session_(\d{3})\.json/)[1]);
            const bNum = Number(b.match(/session_(\d{3})\.json/)[1]);
            return aNum - bNum;
        });
};

const nextSessionSequence = (dateKey) => {
    const files = sessionFilesForDate(dateKey);
    if (files.length === 0) {
        return 1;
    }

    const lastFile = files[files.length - 1];
    const match = /session_(\d{3})\.json/.exec(lastFile);
    if (!match) {
        return 1;
    }
    return Number(match[1]) + 1;
};

const sessionFilePathFromId = (sessionId) => {
    const parsed = parseSessionId(sessionId);
    if (!parsed) {
        return null;
    }

    return path.join(dateFolderPath(parsed.dateKey), buildSessionFileName(parsed.sequence));
};

const readSessionById = (sessionId) => {
    try {
        const filePath = sessionFilePathFromId(sessionId);
        if (!filePath || !fs.existsSync(filePath)) {
            return null;
        }
        return readJsonFile(filePath);
    } catch (error) {
        console.error('Error reading session by id:', error.message);
        return null;
    }
};

const saveSessionById = (session) => {
    const filePath = sessionFilePathFromId(session.id);
    if (!filePath) {
        throw new Error('Session id format is invalid.');
    }
    ensureDir(path.dirname(filePath));
    writeJsonFile(filePath, session);
};

const createSessionFile = (baseSession) => {
    const dateKey = todayDateKey();
    const folderPath = dateFolderPath(dateKey);
    ensureDir(folderPath);

    for (let attempt = 0; attempt < 10; attempt += 1) {
        const sequence = nextSessionSequence(dateKey);
        const fileName = buildSessionFileName(sequence);
        const filePath = path.join(folderPath, fileName);
        const sessionId = buildSessionId(dateKey, sequence);

        const session = {
            ...baseSession,
            id: sessionId,
            date_folder: dateKey,
            sequence,
            file_name: fileName,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
        };

        try {
            writeJsonFile(filePath, session, { flag: 'wx' });
            return session;
        } catch (error) {
            if (error && error.code === 'EEXIST') {
                continue;
            }
            throw error;
        }
    }

    throw new Error('Unable to allocate a new session file. Please retry.');
};

const listSessionsByDate = (dateKey) => {
    const files = sessionFilesForDate(dateKey);
    const folderPath = dateFolderPath(dateKey);

    const sessions = [];
    for (const fileName of files) {
        try {
            const session = readJsonFile(path.join(folderPath, fileName));
            sessions.push({
                session_id: session.id,
                sequence: session.sequence,
                file_name: session.file_name,
                status: session.status,
                model: session.model,
                query: session.query,
                created_at: session.created_at,
                updated_at: session.updated_at,
                history_count: Array.isArray(session.history) ? session.history.length : 0,
            });
        } catch (error) {
            console.error(`Skipping unreadable session file ${fileName}:`, error.message);
        }
    }

    return sessions.sort((a, b) => (b.sequence || 0) - (a.sequence || 0));
};

const callRagService = async (pathName, payload = null) => {
    if (typeof fetch !== 'function') {
        throw new Error('Global fetch is not available. Please run on Node.js 18+');
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), RAG_SERVICE_TIMEOUT_MS);

    try {
        const response = await fetch(`${RAG_SERVICE_URL}${pathName}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: payload ? JSON.stringify(payload) : null,
            signal: controller.signal
        });

        if (!response.ok) {
            const detail = await response.text();
            throw new Error(`Python RAG service failed (${response.status}): ${detail}`);
        }

        return await response.json();
    } finally {
        clearTimeout(timer);
    }
};

const answerWithRag = async (query) => {
    try {
        const payload = await callRagService('/rag/answer', { query });
        return {
            answer: payload.response,
            retrieval: payload.retrieval,
            diagnostics: payload.diagnostics
        };
    } catch (error) {
        console.error('Gateway RAG error:', error.message);
        return {
            answer: 'He thong local LLM tam thoi chua san sang. Vui long kiem tra Python RAG service va Ollama.',
            retrieval: {
                retrieval_status: 'fallback_error',
                top_k: 0,
                snippets: [],
                score_summary: null
            },
            diagnostics: {
                latency_ms: 0,
                context_used: 0,
                error: error.message
            }
        };
    }
};

const buildPromptFromFeedback = (query, feedback = null) => {
    if (!feedback) {
        return query;
    }

    return [
        `Cau hoi goc: ${query}`,
        `Nguoi dung khong dong y voi phan hoi truoc va de xuat: ${feedback}`,
        'Hay tra loi lai sat voi du lieu retrieve hon.'
    ].join('\n');
};

// ============ API ENDPOINTS ============

/**
 * API 1: Tạo session mới
 * POST /chat/start
 */
app.post('/chat/start', async (req, res) => {
    try {
        const { query, model = 'mistral' } = req.body;
        
        if (!query || query.trim() === '') {
            return res.status(400).json({ error: 'Query is required' });
        }
        
        const ragResult = await answerWithRag(query);
        const initialResponse = ragResult.answer;

        const createdSession = createSessionFile({
            model: model,
            query: query,
            status: 'pending',
            history: [
                {
                    llm: initialResponse,
                    client: null,
                    retrieval: ragResult.retrieval,
                    diagnostics: ragResult.diagnostics
                }
            ]
        });
        
        res.json({
            session_id: createdSession.id,
            response: initialResponse,
            retrieval: ragResult.retrieval
        });
    } catch (error) {
        console.error('Failed to start chat session:', error.message);
        res.status(500).json({ error: 'Failed to start chat session' });
    }
});

/**
 * API 2: Gửi feedback (Agree / Disagree)
 * POST /chat/feedback
 */
app.post('/chat/feedback', async (req, res) => {
    try {
        const { session_id, action, feedback } = req.body;
        
        if (!session_id || !action) {
            return res.status(400).json({ error: 'session_id and action are required' });
        }
        
        const session = readSessionById(session_id);
        
        if (!session) {
            return res.status(404).json({ error: 'Session not found' });
        }
        
        if (action === 'agree') {
            // Update last history with client comment
            const lastIndex = session.history.length - 1;
            if (lastIndex >= 0) {
                session.history[lastIndex].client = feedback || 'Agreed (no comment)';
            }
            session.status = 'agreed';
            session.updated_at = new Date().toISOString();

            saveSessionById(session);
            
            res.json({
                status: 'agreed',
                response: null
            });
        } 
        else if (action === 'disagree') {
            // Update last history with client feedback
            const lastIndex = session.history.length - 1;
            if (lastIndex >= 0 && feedback) {
                session.history[lastIndex].client = feedback;
            }
            
            const followupQuery = buildPromptFromFeedback(session.query, feedback);
            const ragResult = await answerWithRag(followupQuery);
            const newResponse = ragResult.answer;
            
            // Add new AI response to history
            session.history.push({
                llm: newResponse,
                client: null,
                retrieval: ragResult.retrieval,
                diagnostics: ragResult.diagnostics
            });
            
            // Status remains pending
            session.status = 'pending';
            session.updated_at = new Date().toISOString();

            saveSessionById(session);
            
            res.json({
                status: 'pending',
                response: newResponse,
                retrieval: ragResult.retrieval
            });
        } 
        else {
            res.status(400).json({ error: 'Invalid action' });
        }
    } catch (error) {
        console.error('Failed to process feedback:', error.message);
        res.status(500).json({ error: 'Failed to process feedback' });
    }
});

/**
 * API 3: List sessions theo ngay
 * GET /chat/sessions?date=DD-MM-YYYY
 */
app.get('/chat/sessions', (req, res) => {
    const requestedDate = String(req.query.date || todayDateKey()).trim();

    if (!parseDateKey(requestedDate)) {
        return res.status(400).json({ error: 'date must be DD-MM-YYYY' });
    }

    try {
        const sessions = listSessionsByDate(requestedDate);
        return res.json({
            date: requestedDate,
            sessions,
        });
    } catch (error) {
        console.error('Failed to list sessions:', error.message);
        return res.status(500).json({ error: 'Failed to list sessions' });
    }
});

/**
 * API 4: Lấy thông tin session
 * GET /chat/:session_id
 */
app.get('/chat/:session_id', (req, res) => {
    const { session_id } = req.params;
    const session = readSessionById(session_id);
    
    if (!session) {
        return res.status(404).json({ error: 'Session not found' });
    }
    
    res.json(session);
});

/**
 * API 5: Trigger ingest on Python RAG service
 * POST /rag/ingest
 */
app.post('/rag/ingest', async (req, res) => {
    try {
        const result = await callRagService('/rag/ingest');
        res.json(result);
    } catch (error) {
        console.error('Failed to trigger Python ingest:', error.message);
        res.status(500).json({ error: 'Failed to trigger Python ingest' });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
});