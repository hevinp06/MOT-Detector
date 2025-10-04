from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from msal import ConfidentialClientApplication, SerializableTokenCache
import requests
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List
import re
import threading

load_dotenv()

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
API_KEY = os.getenv("API_key")


DVLA_API_URL = "https://history.mot.api.gov.uk/v1/trade/vehicles/registration"
DVLA_SCOPE = ["https://tapi.dvsa.gov.uk/.default"]

token_cache = SerializableTokenCache()
token_cache_lock = threading.Lock()

msal_app = None

def initialize_msal_app():
    global msal_app
    with token_cache_lock:
        if msal_app is None:
            msal_app = ConfidentialClientApplication(
                client_id=CLIENT_ID,
                client_credential=CLIENT_SECRET,
                authority=f"https://login.microsoftonline.com/{TENANT_ID}",
                token_cache=token_cache
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_msal_app()  
    
    yield 

app = FastAPI(lifespan=lifespan)

def get_access_token() -> str:
    """Retrieve an access token using MSAL with client credentials flow."""
    with token_cache_lock:
        if msal_app is None:
            initialize_msal_app()
        try:
            result = msal_app.acquire_token_silent(scopes=DVLA_SCOPE, account=None)
            if not result:
                result = msal_app.acquire_token_for_client(scopes=DVLA_SCOPE)
            if "access_token" in result:
                return result["access_token"]
            else:
                error_description = result.get("error_description", "Unknown error")
                raise HTTPException(status_code=500, detail=f"Could not obtain access token: {error_description}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Token acquisition failed: {str(e)}")

class Defect(BaseModel):
    text: str
    type: str
    dangerous: bool

class MOTTest(BaseModel):
    registrationAtTimeOfTest: Optional[str] = None
    completedDate: str
    testResult: str
    expiryDate: Optional[str] = None
    odometerValue: str
    odometerUnit: str
    odometerResultType: str
    motTestNumber: str
    dataSource: str
    location: Optional[str] = None
    defects: Optional[List[Defect]] = None

class VehicleResponse(BaseModel):
    registration: str
    make: Optional[str] = None
    model: Optional[str] = None
    firstUsedDate: Optional[str] = None
    fuelType: Optional[str] = None
    primaryColour: Optional[str] = None
    registrationDate: Optional[str] = None
    manufactureDate: Optional[str] = None
    engineSize: Optional[str] = None
    hasOutstandingRecall: Optional[str] = None
    motTests: Optional[List[MOTTest]] = None

class ErrorResponse(BaseModel):
    errorCode: str
    errorMessage: str
    requestId: str

@app.get("/mot/", include_in_schema=False)
async def mot_root():
    raise HTTPException(
        status_code=400,
        detail="Invalid vehicle registration number"
    )

@app.get("/mot/{plate_number}", response_model=VehicleResponse)
async def get_mot(plate_number: str):
    """Fetch MOT and vehicle details for a given registration number."""
    # Validate plate number
    plate_number = plate_number.strip().upper()
    if not re.match(r"^[A-Z0-9\s]{1,7}$", plate_number):
        raise HTTPException(status_code=400, detail="Invalid UK registration number format")

    access_token = get_access_token()

    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    url = f"{DVLA_API_URL}/{plate_number}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        return VehicleResponse(**data)

    except requests.exceptions.HTTPError as http_err:
        try:
            error_data = response.json()
            error_response = ErrorResponse(**error_data)
            error_message = f"DVLA API error: {error_response.errorMessage} (Code: {error_response.errorCode}, Request ID: {error_response.requestId})"
        except ValueError:
            error_message = response.text
        if response.status_code == 400:
            raise HTTPException(status_code=400, detail=f"Invalid request: {error_message}")
        elif response.status_code == 403:
            raise HTTPException(status_code=403, detail="Access to DVLA API denied. Check API key or permissions.")
        elif response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Vehicle with registration {plate_number} not found.")
        elif response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        else:
            raise HTTPException(status_code=response.status_code, detail=error_message)
    except requests.exceptions.RequestException as req_err:
        raise HTTPException(status_code=500, detail=f"Network error: {str(req_err)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")