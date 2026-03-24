const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3000;

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

// Helper: Generate AI response (mock)
const generateAIResponse = (query, feedback = null) => {
    if (feedback) {
        return `Cảm ơn phản hồi của bạn: "${feedback}". Đây là câu trả lời đã được cập nhật dựa trên góp ý của bạn.`;
    }
    return `Đây là câu trả lời cho câu hỏi: "${query}". Bạn có đồng ý với câu trả lời này không?`;
};

// ============ API ENDPOINTS ============

/**
 * API 1: Tạo session mới
 * POST /chat/start
 */
app.post('/chat/start', (req, res) => {
    const { query, model = 'deepseek' } = req.body;
    
    if (!query || query.trim() === '') {
        return res.status(400).json({ error: 'Query is required' });
    }
    
    const sessionId = Date.now().toString() + '-' + Math.random().toString(36).substr(2, 6);
    const initialResponse = generateAIResponse(query);
    
    const sessions = readSessions();
    sessions[sessionId] = {
        id: sessionId,
        model: model,  // <--- THÊM DÒNG NÀY
        query: query,
        status: 'pending',
        history: [
            {
                llm: initialResponse,
                client: null
            }
        ]
    };
    
    writeSessions(sessions);
    
    res.json({
        session_id: sessionId,
        response: initialResponse
    });
});

/**
 * API 2: Gửi feedback (Agree / Disagree)
 * POST /chat/feedback
 */
app.post('/chat/feedback', (req, res) => {
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
        
        // Generate new response based on feedback
        const newResponse = generateAIResponse(session.query, feedback);
        
        // Add new AI response to history
        session.history.push({
            llm: newResponse,
            client: null
        });
        
        // Status remains pending
        session.status = 'pending';
        
        writeSessions(sessions);
        
        res.json({
            status: 'pending',
            response: newResponse
        });
    } 
    else {
        res.status(400).json({ error: 'Invalid action' });
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

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
});