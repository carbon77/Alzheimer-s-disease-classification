package com.zakat.classifier

data class PredictionResult(
    val probabilities: List<Float>,
    val predictedClass: Int,
    val predictedClassTitle: String,
)
