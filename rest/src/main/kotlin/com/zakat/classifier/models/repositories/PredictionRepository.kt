package com.zakat.classifier.models.repositories

import com.zakat.classifier.models.Prediction
import org.springframework.data.repository.CrudRepository
import org.springframework.stereotype.Repository
import java.util.*

@Repository
interface PredictionRepository : CrudRepository<Prediction, UUID> {
}