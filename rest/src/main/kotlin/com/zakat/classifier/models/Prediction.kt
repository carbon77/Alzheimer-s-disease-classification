package com.zakat.classifier.models

import org.springframework.data.annotation.Id
import org.springframework.data.relational.core.mapping.Table
import java.time.Instant
import java.util.UUID

@Table(name = "predictions")
data class Prediction(
    var s3Key: String = "",
    var logits: FloatArray = floatArrayOf(),
    var probabilities: FloatArray = floatArrayOf(),
    var predictedClass: Int? = null,
    var createdAt: Instant = Instant.now(),
    @Id
    var predictionId: UUID? = null,
)