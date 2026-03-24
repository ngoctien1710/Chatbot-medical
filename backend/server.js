const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = Number(process.env.PORT || 3000);
const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL || 'http://127.0.0.1:8001';
const RAG_SERVICE_TIMEOUT_MS = Number(process.env.RAG_SERVICE_TIMEOUT_MS || 120000);

// Middleware
app.use(cors());
app.use(express.json());

// File to store sessions
const SESSIONS_FILE = path.join(__dirname, 'sessions.json');

// Helper: Read sessions from file
const readSessions = () => {
    try {
        if (!fs.existsSync(SESSIONS_FILE)) {
            return {};
        }
        const data = fs.readFileSync(SESSIONS_FILE, 'utf8');
        return JSON.parse(data);
    } catch (error) {
        console.error('Error reading sessions:', error);
        return {};
    }
};

// Helper: Write sessions to file
const writeSessions = (sessions) => {
    try {
        fs.writeFileSync(SESSIONS_FILE, JSON.stringify(sessions, null, 2));
        console.log('✅ Sessions saved');
    } catch (error) {
        console.error('Error writing sessions:', error);
    }
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
        const { query, model = 'deepseek' } = req.body;
        
        if (!query || query.trim() === '') {
            return res.status(400).json({ error: 'Query is required' });
        }
        
        const sessionId = Date.now().toString() + '-' + Math.random().toString(36).substr(2, 6);
        const ragResult = await answerWithRag(query);
        const initialResponse = ragResult.answer;
        
        const sessions = readSessions();
        sessions[sessionId] = {
            id: sessionId,
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
        };
        
        writeSessions(sessions);
        
        res.json({
            session_id: sessionId,
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
        
        const sessions = readSessions();
        const session = sessions[session_id];
        
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
            
            writeSessions(sessions);
            
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
            
            writeSessions(sessions);
            
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
 * API 3: Lấy thông tin session
 * GET /chat/:session_id
 */
app.get('/chat/:session_id', (req, res) => {
    const { session_id } = req.params;
    
    const sessions = readSessions();
    const session = sessions[session_id];
    
    if (!session) {
        return res.status(404).json({ error: 'Session not found' });
    }
    
    res.json(session);
});

/**
 * API 4: Trigger ingest on Python RAG service
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