package com.zakat.classifier

import com.zakat.classifier.models.Prediction
import com.zakat.classifier.services.PredictionService
import org.springframework.web.bind.annotation.*
import org.springframework.web.multipart.MultipartFile
import java.util.*

@RestController
@RequestMapping("/predict")
class ClassifierController(
    private val predictionService: PredictionService,
) {

    @PostMapping
    fun runModel(@RequestParam("image") image: MultipartFile): Prediction {
        return predictionService.predictClass(image)
    }

    @GetMapping("all")
    fun findAll(): Iterable<Prediction> = predictionService.findAll()

    @DeleteMapping("{predictionId}")
    fun deletePrediction(@PathVariable("predictionId") predictionId: UUID) {
        predictionService.deletePrediction(predictionId)
    }
}