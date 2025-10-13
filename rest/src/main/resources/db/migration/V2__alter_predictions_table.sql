ALTER TABLE predictions
    RENAME COLUMN image_url TO s3_key;
ALTER TABLE predictions
    ADD COLUMN logits NUMERIC[];