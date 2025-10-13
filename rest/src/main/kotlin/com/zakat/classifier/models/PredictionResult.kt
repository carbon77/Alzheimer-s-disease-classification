package com.zakat.classifier.models

data class PredictionResult(
    val probabilities: FloatArray,
    val predictedClass: Int,
)