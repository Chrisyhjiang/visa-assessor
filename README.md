# O-1A Visa Assessor

An AI-powered tool that assesses eligibility for O-1A visa applications by analyzing CVs and resumes.

![O-1A Visa Assessor Screenshot](frontend/public/screenshot.png)

## Features

- Upload your CV/resume in PDF, DOCX, or TXT format
- AI-powered assessment of O-1A visa criteria matches
- Detailed breakdown of qualification with evidence
- Personalized recommendations to improve your application
- Modern, responsive UI for desktop and mobile

## Tech Stack

- **Frontend**: React with TypeScript
- **Backend**: FastAPI (Python)
- **Containerization**: Docker

## Getting Started

### Prerequisites

- Docker and Docker Compose

### Running the Application

1. Clone the repository
2. Create a `.env` file with required environment variables
3. Start the application:

```bash
docker-compose up
```

4. Open your browser and go to `http://localhost:3000`

## API Documentation

API documentation is available at `http://localhost:8000/docs` when the application is running.

## Development

### Frontend Development

```bash
cd frontend
npm install
npm start
```

### Backend Development

```bash
cd app
pip install -r requirements.txt
python -m uvicorn app.app:app --reload
```

## License

MIT License

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [Docker](https://www.docker.com/)
