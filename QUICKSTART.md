# 🎨 Air Canvas - Quick Start Guide

## ⚡ Get Started in 30 Seconds

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Application
```bash
python main.py
```

### 3️⃣ Calibrate Your Hand
- **Press SPACEBAR**
- Put your hand in the box shown on screen
- Wait for "Skin color calibrated!" message

### 4️⃣ Start Drawing!
- **Point with Index Finger** → Draw
- **Hold 3+ Fingers** → Erase
- **Press C** → Clear Canvas
- **Press S** → Save Your Drawing

---

## 🎮 All Controls

| Action | Key |
|--------|-----|
| Calibrate | SPACEBAR |
| Draw | Index Finger |
| Erase | 3+ Fingers |
| Clear | C |
| Save | S |
| Thinner | Z |
| Thicker | X |
| Prev Color | Q |
| Next Color | W |
| Exit | ESC |

---

## 🎨 Available Colors

1. **Green** (default)
2. **Blue**
3. **Red**
4. **Yellow**
5. **Purple**
6. **Cyan**
7. **White**
8. **Black**

---

## 📋 Features

✅ Real-time hand tracking  
✅ Smooth drawing with 8 colors  
✅ Adjustable brush (3-15px)  
✅ Save drawings as PNG  
✅ Clear canvas with one key  
✅ FPS counter  
✅ Error handling  

---

## ⚠️ First-Time Setup

1. Ensure **good lighting** in your room
2. **Calibrate skin color** first (press SPACEBAR)
3. Keep hand **clear and visible** in frame
4. Stay **30-60cm** from camera

---

## 💡 Tips

- **Can't detect hand?** → Better lighting or recalibrate (SPACEBAR)
- **Drawing not smooth?** → Increase thickness (press X)
- **Want more precision?** → Slow down hand movements
- **Colors not showing?** → Try Q/W to cycle colors

---

## 📸 Saving Your Artwork

Drawings are automatically saved to PNG files with timestamps:
- `drawing_20260222_120530.png`
- Find them in the project folder

---

## 🐛 Troubleshooting

### Issue: Application won't start
**Solution:** Check if camera is connected and not in use by another app

### Issue: Hand not detected
**Solution:** Press SPACEBAR to recalibrate, ensure good lighting

### Issue: Laggy performance
**Solution:** Close background apps, reduce resolution in config.py

### Issue: Colors look wrong
**Solution:** Press Q or W to cycle to correct color

---

## 🎓 About This Project

**Air Canvas** is a professional, resume-level application that demonstrates:
- Computer vision with OpenCV
- Real-time hand gesture recognition
- Object-oriented Python design
- Production-quality code architecture

---

## 📞 Support

For detailed information, see:
- [README.md](README.md) - Full documentation
- [SETUP_SUMMARY.md](SETUP_SUMMARY.md) - Technical overview
- [config.py](config.py) - Configuration options

---

**Ready to draw? Run `python main.py` now!** 🚀
