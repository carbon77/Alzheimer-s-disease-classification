package com.zakat.classifier.services

import com.zakat.classifier.models.Prediction
import com.zakat.classifier.models.repositories.PredictionRepository
import org.springframework.stereotype.Service
import org.springframework.web.multipart.MultipartFile

@Service
class PredictionService(
    private val modelService: ModelService,
    private val predictionRepository: PredictionRepository,
) {

    fun predictClass(image: MultipartFile): Prediction {
        val result = modelService.runModel(image)
        var prediction = Prediction(
            probabilities = result.probabilities,
            predictedClass = result.predictedClass,
        )
        prediction = predictionRepository.save(prediction)
        return prediction
    }

    fun findAll(): Iterable<Prediction> {
        return predictionRepository.findAll()
    }
}