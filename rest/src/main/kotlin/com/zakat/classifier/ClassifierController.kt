package com.zakat.classifier

import com.zakat.classifier.models.Prediction
import com.zakat.classifier.models.PredictionResult
import com.zakat.classifier.models.repositories.PredictionRepository
import com.zakat.classifier.services.ModelService
import com.zakat.classifier.services.PredictionService
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile

@RestController
@RequestMapping("/predict")
class ClassifierController(
    private val predictionService: PredictionService,
) {

    @PostMapping
    fun runModel(@RequestParam("image") image: MultipartFile): Prediction {
        return predictionService.predictClass(image)
    }

    @GetMapping("/all")
    fun findAll(): Iterable<Prediction> = predictionService.findAll()
}