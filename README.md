# Credit Default Prediction System

This project implements a credit default prediction system using machine learning, with a REST API and Docker Swarm deployment.

## Project Structure

```
.
├── api/
│   └── app.py              # FastAPI application
├── model/
│   └── train.py           # Model training script
├── data/
│   └── cs-training.csv    # Dataset (not included, download from Kaggle)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup Instructions

1. Download the Give Me Some Credit dataset from Kaggle and place it in the `data` directory as `cs-training.csv`

2. Train the model:
```bash
python model/train.py
```

3. Initialize Docker Swarm:
```bash
docker swarm init
```

4. Build and deploy the application:
```bash
docker-compose build
docker stack deploy -c docker-compose.yml credit-prediction
```

## API Usage

The API will be available at `http://localhost:8000`

### Endpoints

- GET `/`: Health check endpoint
- POST `/predict`: Make predictions

Example prediction request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "RevolvingUtilizationOfUnsecuredLines": 0.5,
           "age": 45,
           "NumberOfTime30_59DaysPastDueNotWorse": 0,
           "DebtRatio": 0.1,
           "MonthlyIncome": 5000,
           "NumberOfOpenCreditLinesAndLoans": 2,
           "NumberOfTimes90DaysLate": 0,
           "NumberRealEstateLoansOrLines": 1,
           "NumberOfTime60_89DaysPastDueNotWorse": 0,
           "NumberOfDependents": 2
         }'
```

## Monitoring

To check service status:
```bash
docker service ls
docker service ps credit-prediction_api
```

## Scaling

To scale the service:
```bash
docker service scale credit-prediction_api=5
```