const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const admin = require("firebase-admin");
const serviceAccount = require("./firebase_key.json"); // ตรวจสอบชื่อไฟล์ Key ให้ตรง

// Initialize Firebase
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: "https://security-camera-c2be0-default-rtdb.asia-southeast1.firebasedatabase.app/" // ใส่ URL ของคุณ
});
const db = admin.database();

const app = express();
app.use(cors());
app.use(bodyParser.json());

// รับข้อมูลจาก Python
app.post('/api/incident', async (req, res) => {
    try {
        await db.ref('incidents').push(req.body);
        console.log("🔥 Logged:", req.body.class);
        res.json({ status: "success" });
    } catch (error) {
        console.error(error);
        res.status(500).send(error);
    }
});

// ส่งข้อมูลให้ React
app.get('/api/logs', async (req, res) => {
    try {
        const snapshot = await db.ref('incidents').limitToLast(50).once('value');
        const data = snapshot.val();
        const logs = data ? Object.values(data).reverse() : [];
        res.json(logs);
    } catch (error) {
        res.status(500).send(error);
    }
});

// Login (Mockup)
app.post('/api/login', (req, res) => {
    const { username, password } = req.body;
    if (username === "admin" && password === "admin") {
        res.json({ success: true });
    } else {
        res.status(401).json({ success: false });
    }
});

app.listen(3000, () => console.log('🟢 Backend running on port 3000'));