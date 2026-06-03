# Rift Draft: Frontend Client

This directory contains the React/TypeScript/Vite-based client application for **Rift Draft**. For detailed architecture, backend setup, environment configurations, and full-stack integration instructions, please refer to the primary [Root README](../README.md).

---

## Tech Stack

- **Framework**: React 19
- **Build Tool**: Vite
- **Language**: TypeScript
- **Styling**: Vanilla CSS with relative viewport aspect ratio responsive scaling

---

## Quick Start (Development)

### 1. Configure Environments
Create a `.env` file in the root of this `frontend` directory:

```ini
VITE_WINRATE_API_URL="http://localhost:8000/api/v1/predict"
VITE_COACHING_API_URL="http://localhost:8000/api/v1/agent"
```

### 2. Install and Start

```bash
# Install dependencies
npm install

# Start the local development server
npm run dev
```

---

## Production Builds

To compile a production bundle:

```bash
# Type check and build distribution assets
npm run build

# Preview production build locally
npm run preview
```
The compiled files will be located in the `dist` directory.
