package com.zakat.classifier

import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile

@RestController
class ClassifierController(
    private val classifierService: ClassifierService,
) {

    @PostMapping("/run")
    fun runModel(@RequestParam("image") image: MultipartFile): PredictionResult {
        return classifierService.runModel(image)
    }
}