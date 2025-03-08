import uvicorn

if __name__ == "__main__":
    print("Starting O-1A Visa Qualification Assessment API...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 