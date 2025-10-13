CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    image_url TEXT,
    probabilities NUMERIC[] NOT NULL,
    predicted_class INTEGER,
    created_at TIMESTAMP DEFAULT now()
);