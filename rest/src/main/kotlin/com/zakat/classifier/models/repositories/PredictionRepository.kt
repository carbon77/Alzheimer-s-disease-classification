package com.zakat.classifier.models.repositories

import com.zakat.classifier.models.Prediction
import org.springframework.data.repository.CrudRepository
import org.springframework.stereotype.Repository

@Repository
interface PredictionRepository : CrudRepository<Prediction, String> {
}