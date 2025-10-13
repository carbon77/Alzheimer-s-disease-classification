package com.zakat.classifier.models

data class PredictionResult(
    val logits: FloatArray,
    val probabilities: FloatArray,
    val predictedClass: Int,
)