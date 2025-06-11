from fastapi import FastAPI,UploadFile,File
from fastapi.responses import JSONResponse,StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from .detect import detect_defects

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    Root endpoint to check if the server is running.
    
    Returns:
    JSONResponse: A simple message indicating the server is running.
    """
    return JSONResponse(content={"message": "Defect Detection API is running"})
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Endpoint to detect defects in an uploaded image."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"message": "File must be an image"}
            )
        
        # Read and process image
        contents = await file.read()
        img = Image.open(BytesIO(contents))
        
        # Detect defects
        result_img = detect_defects(img)
        
        # Return processed image
        img_byte_arr = BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr, 
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=result.png"}
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"message": f"Processing error: {str(e)}"}
        )