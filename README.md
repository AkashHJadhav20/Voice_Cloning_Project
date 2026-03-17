# 🎤 Voice Cloning Project - XTTS-v2

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A professional **voice cloning Text-to-Speech** system using **XTTS-v2** with support for **4 speakers (AB, AP, ME, DF)**. Features multiple interfaces (CLI, Flask Web UI, Gradio) and advanced audio enhancement for maximum voice similarity.

## ✨ Features

- 🎯 **4 Unique Voices** - Pre-configured for speakers AB, AP, ME, DF
- 🚀 **GPU Optimized** - Runs on GTX 1650 (4GB VRAM) with memory optimizations
- 🎚️ **Professional Audio Enhancement** - Noise reduction, de-essing, loudness normalization
- 🌐 **3 Interface Options** - CLI, Flask Web UI, and Modern Gradio Interface
- 📊 **Voice Quality Metrics** - PESQ, STOI, speaker verification scores
- 💾 **Persistent Voice Selection** - Remembers your preferred voice
- 🔄 **Generation History** - Automatically saves all generated audio
- 📥 **Easy Download** - Download generated audio with one click

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Voice Enhancement](#-voice-enhancement)
- [Troubleshooting](#-troubleshooting)
- [Technical Details](#-technical-details)
- [License](#-license)

## 💻 System Requirements

### Minimum Hardware
| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA GTX 1650 (4GB VRAM) or equivalent |
| **CPU** | Intel i5 / AMD Ryzen 5 |
| **RAM** | 8GB |
| **Storage** | 10GB free space |
| **OS** | Windows 10/11, Linux, macOS (CUDA only on NVIDIA) |

### Software Requirements
- **Python** 3.8 - 3.10 (3.10 recommended)
- **CUDA Toolkit** 11.8 (for GPU acceleration)
- **NVIDIA Driver** 525.60.13 or newer

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/voice-cloning-project.git
cd voice-cloning-project
