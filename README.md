# Lab 1: First ML Product - Credit Default Risk Scoring API

## Project Description and Features

This lab delivers a **Credit Default Risk Scoring API** for **DDM501 - AI in DevOps, DataOps, MLOps**. A trained **HistGradientBoostingClassifier** estimates a credit card customer's probability of defaulting on their next payment. The service converts that probability into an **APPROVE**, **REVIEW**, or **DECLINE** decision.

Training produces `models/credit_model.joblib`, containing a preprocessing and classification pipeline plus training metadata. The API loads the artifact once at startup.

### Features

- **Model training and persistence:** trains on the included synthetic credit dataset, evaluates an 80/20 stratified split, and saves the pipeline and metrics.
- **REST API:** health checks, single predictions, batches of up to 500 applications, model metadata, and interactive Swagger documentation.
- **Input validation and error handling:** Pydantic constraints and custom validators return 422 for malformed input; unavailable models return 503 from prediction endpoints.
- **Configurable decisions:** environment variables control review and decline thresholds, which are included in each scoring response.
- **Container deployment:** a non-root Docker image and Compose service with a read-only model mount and readiness health check.
- **Testing and coverage:** pytest checks API behavior, validation, representative model behavior, and batch consistency; GitHub Actions trains, tests, builds, and smoke-tests the container.

## Project Structure

```text
ddm501-lab1/
├── app/
│   ├── config.py           # Environment configuration and decision thresholds
│   ├── main.py             # FastAPI routes and startup model loading
│   ├── model.py            # Artifact loading, feature mapping, and scoring
│   └── schemas.py          # Pydantic request and response contracts
├── data/credit_default.csv # Included generated dataset
├── models/                # Training output: credit_model.joblib
├── scripts/
│   ├── make_dataset.py    # Regenerate synthetic data
│   ├── download_data.py   # Optional UCI dataset download
│   └── train_model.py     # Train, evaluate, and save the pipeline
├── tests/test_api.py      # API, validation, and model behavior tests
├── .github/workflows/     # Install, train, test, and container smoke checks
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
└── requirements.txt
```

## Prerequisites

- Python 3.11, matching the Docker image and CI workflow
- Git
- Docker and Docker Compose for container deployment (Docker Desktop provides both on Windows and macOS)

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/NamTe/msa36hn-ddm501-lab1.git
cd msa36hn-ddm501-lab1

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate.bat   # Windows Command Prompt

# Install dependencies
python -m pip install -r requirements.txt
```

Run subsequent commands from the directory containing `requirements.txt`.

### 2. Train the Model

```bash
python scripts/train_model.py
```

Training uses the included CSV, prints evaluation metrics, and writes `models/credit_model.joblib`. No dataset download is required. Train before starting the API or running tests, and restart the API after replacing the artifact.

### 3. Run the API

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [Swagger UI](http://localhost:8000/docs) or [ReDoc](http://localhost:8000/redoc) to explore the API.

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Score one applicant
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"limit_bal":300000,"sex":2,"education":1,"marriage":2,"age":38,
       "pay_status":[-1,-1,-1,-1,-1,-1],
       "bill_amt":[12000,11500,11000,10500,10000,9500],
       "pay_amt":[12000,11500,11000,10500,10000,9500]}'
```

### 5. Run with Docker

After training the model:

```bash
docker compose up --build -d
docker compose ps
docker inspect --format='{{.State.Health.Status}}' credit-risk-api
```

Compose exposes port `8000` and mounts `./models:/app/models:ro`. The artifact is not baked into the image. The container runs as `appuser` and checks that `/health` reports `model_loaded: true`.

```bash
# View logs
docker compose logs -f api

# Reload a newly trained artifact
docker compose restart api

# Stop the service
docker compose down
```

## Running Tests

Install dependencies and train the model before running the suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with terminal and HTML coverage reports
python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html
```

Open `htmlcov/index.html` for the HTML report. The suite covers health and metadata, prediction fields, deterministic scores, threshold consistency, representative risk-ordering behavior, invalid inputs, batch ordering, and agreement between single and batch scoring. The repayment-delay example does not guarantee global model monotonicity.

The GitHub Actions smoke workflow installs dependencies on Python 3.11, verifies wheel availability and dataset shape, trains the model, runs tests with coverage, builds the image, and checks health and prediction in a running container.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Service metadata and links |
| GET | `/health` | Model readiness and version |
| POST | `/predict` | Score one credit application |
| POST | `/predict/batch` | Score 1 to 500 applications in input order |
| GET | `/model/info` | Model type, training metadata, metrics, and thresholds |
| GET | `/docs` | Interactive Swagger documentation |
| GET | `/redoc` | ReDoc documentation |

## API Usage Examples

The examples assume the API is running at `http://localhost:8000`. Prediction probabilities below are illustrative; actual scores depend on the trained artifact.

### Health Check (`GET /health`)

```bash
curl -i http://localhost:8000/health
```

Example response body (200):

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

This endpoint also returns HTTP 200 when the model is unavailable, with `status: "unhealthy"` and `model_loaded: false`. Inspect the body to determine readiness.

### Model Info (`GET /model/info`)

```bash
curl http://localhost:8000/model/info
```

The response includes `model_version`, `model_type`, `trained_at`, `metrics`, `review_threshold`, `decline_threshold`, and `is_loaded`. Metrics come from the loaded artifact and include ROC AUC, PR AUC, confusion-matrix counts, and the training evaluation threshold. Training metadata is unavailable when no model was loaded.

### Single Prediction (`POST /predict`)

Paste this payload into Swagger UI's `POST /predict`, or save it as `application.json` in your current directory:

```json
{
  "limit_bal": 300000,
  "sex": 2,
  "education": 1,
  "marriage": 2,
  "age": 38,
  "pay_status": [-1, -1, -1, -1, -1, -1],
  "bill_amt": [12000, 11500, 11000, 10500, 10000, 9500],
  "pay_amt": [12000, 11500, 11000, 10500, 10000, 9500]
}
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary @application.json
```

Illustrative response only; the probability depends on the trained artifact:

```json
{
  "default_probability": 0.12,
  "risk_band": "LOW",
  "decision": "APPROVE",
  "review_threshold": 0.3,
  "decline_threshold": 0.6,
  "model_version": "1.0.0"
}
```

### Batch Prediction (`POST /predict/batch`)

Wrap one or more complete application objects in an `applications` list. For example:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"applications":[{"limit_bal":300000,"sex":2,"education":1,"marriage":2,"age":38,"pay_status":[-1,-1,-1,-1,-1,-1],"bill_amt":[12000,11500,11000,10500,10000,9500],"pay_amt":[12000,11500,11000,10500,10000,9500]}]}'
```

The response contains `predictions` (one six-field result per application) and `total_count`. The whole batch is scored in one vectorized model call. Empty batches and batches larger than 500 fail validation.

Example response body (200) for the one-application batch above:

```json
{
  "predictions": [
    {
      "default_probability": 0.12,
      "risk_band": "LOW",
      "decision": "APPROVE",
      "review_threshold": 0.3,
      "decline_threshold": 0.6,
      "model_version": "1.0.0"
    }
  ],
  "total_count": 1
}
```

### Validation Error: Missing Required Fields (422)

```bash
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{}'
```

The response's `detail` list identifies every missing required field, including `limit_bal`, `sex`, `education`, `marriage`, `age`, and the three history lists.

### Validation Error: Negative Payment (422)

```bash
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"limit_bal":300000,"sex":2,"education":1,"marriage":2,"age":38,
       "pay_status":[-1,-1,-1,-1,-1,-1],
       "bill_amt":[12000,11500,11000,10500,10000,9500],
       "pay_amt":[-1,11500,11000,10500,10000,9500]}'
```

The custom payment validator rejects the negative value in `pay_amt`.

### Validation Error: Empty Batch (422)

```bash
curl -i -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"applications": []}'
```

### Unknown Route (404)

```bash
curl -i http://localhost:8000/predict/does-not-exist
```

Prediction endpoints return `503` with `{"detail":"Model is not loaded"}` when the model is unavailable. Unexpected scoring exceptions are logged and return `500` with `{"detail":"Prediction failed"}`.

## Input Validation and Decision Rules

All fields are required. Monetary amounts use NT dollars. Each six-item history is ordered from the most recent month (`t-1`) to the oldest (`t-6`).

| Field | Type | Accepted values |
| --- | --- | --- |
| `limit_bal` | Number | Greater than 0 and at most 2,000,000 |
| `sex` | Integer | `1` male, `2` female |
| `education` | Integer | `1` graduate school, `2` university, `3` high school, `4` others |
| `marriage` | Integer | `1` married, `2` single, `3` others |
| `age` | Integer | 18 to 100 inclusive |
| `pay_status` | List of integers | Exactly six values, each from -2 to 8 |
| `bill_amt` | List of numbers | Exactly six values; no nonnegative constraint in the schema |
| `pay_amt` | List of numbers | Exactly six nonnegative values |

Repayment status means `-2` no consumption, `-1` paid in full, `0` revolving credit, and `1` through `8` months of payment delay. The wrapper maps `pay_status` to `PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6`; there is no `PAY_1` in the serving feature list. Preserving the 23-column training order avoids training-serving skew.

### Decisions

| Default probability `p` | Risk band | Decision |
| --- | --- | --- |
| `p < 0.30` | `LOW` | `APPROVE` |
| `0.30 <= p < 0.60` | `MEDIUM` | `REVIEW` |
| `p >= 0.60` | `HIGH` | `DECLINE` |

These are the default thresholds; environment configuration can change them. Decisions use the unrounded probability, while `default_probability` is returned rounded to four decimal places. Near a threshold, rounding can therefore make the displayed score appear to be on a different side of the boundary.

## Configuration

Set environment variables before starting the process:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `<repository>/models/credit_model.joblib` | Artifact loaded by the API |
| `MODEL_VERSION` | `1.0.0` | Version label returned in responses |
| `REVIEW_THRESHOLD` | `0.30` | Minimum probability for human review |
| `DECLINE_THRESHOLD` | `0.60` | Minimum probability for decline |

For example, in Bash:

```bash
export REVIEW_THRESHOLD=0.35
export DECLINE_THRESHOLD=0.65
python -m uvicorn app.main:app --reload
```

Keep `0 <= REVIEW_THRESHOLD < DECLINE_THRESHOLD <= 1`; the configuration module does not enforce this ordering. Changing serving thresholds does not change the training script's fixed evaluation threshold. `MODEL_VERSION` is a configured label, not automatically derived from the artifact.

Although `config.py` also defines `HOST`, `PORT`, and `DEBUG`, the launch code does not use them to configure Uvicorn. Use explicit CLI options such as `--host 0.0.0.0 --port 8000`. The application does not automatically load a `.env` file.

For Compose, edit the service's environment entries and run `docker compose up -d` to recreate it.

## Data and Training

The lab includes a generated dataset with 30,000 rows, 23 input features, and the binary target `default_payment_next_month` (`1` means default). Its schema follows the UCI *Default of Credit Card Clients* dataset used by the optional download script; the bundled data is synthetic.

The training script uses an 80/20 stratified split with random seed `501`. A scikit-learn pipeline one-hot encodes `SEX`, `EDUCATION`, and `MARRIAGE`, passes through the remaining numeric features, and fits a `HistGradientBoostingClassifier` with 300 maximum iterations, learning rate `0.06`, maximum depth `6`, and L2 regularization `1.0`.

It prints ROC AUC, average precision (reported as PR AUC), a confusion matrix, and a classification report. Evaluation uses a fixed threshold of `0.30`. The saved joblib bundle contains the pipeline and metadata, including training time, feature names, split sizes, and metrics. Check your training output or `/model/info` for actual results.

To regenerate the synthetic CSV, replacing the existing file:

```bash
python scripts/make_dataset.py --rows 30000 --seed 501
python scripts/train_model.py
```

To optionally replace it with the real UCI data, using network access:

```bash
python -m pip install ucimlrepo openpyxl
python scripts/download_data.py
python scripts/train_model.py
```

Restart the API after retraining: the service loads the artifact once at startup.

## Troubleshooting

- **Unhealthy service or prediction returns 503:** run `python scripts/train_model.py`, verify `MODEL_PATH`, then restart the API. For Compose, ensure the host's `models/credit_model.joblib` exists and is readable by the container user.
- **Dataset missing:** run `python scripts/make_dataset.py` before training.
- **Request returns 422:** check required fields, categorical values, age and credit-limit bounds, history lengths, and payment/status validators.
- **Port 8000 is in use:** choose a different Uvicorn `--port`, or change the host side of the Compose port mapping (for example, `8001:8000`).

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Docker Documentation](https://docs.docker.com/)
- [pytest Documentation](https://docs.pytest.org/)
