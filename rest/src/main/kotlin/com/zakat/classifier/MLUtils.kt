package com.zakat.classifier

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import org.springframework.web.multipart.MultipartFile
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.nio.FloatBuffer
import javax.imageio.ImageIO
import kotlin.math.exp

fun argmax(nums: FloatArray): Int {
    var argMax = 0
    for (i in nums.indices) {
        if (nums[i] > nums[argMax]) {
            argMax = i
        }
    }

    return argMax
}

fun softmax(logits: FloatArray): FloatArray {
    val expSum = logits
        .map { logit -> exp(logit) }
        .sum()

    return logits.map { logit -> exp(logit) / expSum }.toFloatArray()
}

fun convertImageToTensor(env: OrtEnvironment, image: MultipartFile): OnnxTensor =
    image.inputStream.use { stream ->
        val bufferedImage = ImageIO.read(stream)
        val data = preprocessImage(bufferedImage)
        val buffer = FloatBuffer.wrap(data)

        val shape = longArrayOf(1, 3, 200, 200)
        OnnxTensor.createTensor(env, buffer, shape)
    }

fun preprocessImage(image: BufferedImage): FloatArray {
    val resizedImage = resizeImage(image, 200, 200)
    val width = resizedImage.width
    val height = resizedImage.height

    val data = FloatArray(3 * 200 * 200)
    val pixels = resizedImage.getRGB(0, 0, width, height, null, 0, width)

    for (i in 0 until pixels.size) {
        val pixel = pixels[i]
        var r = ((pixel shr 16) and 0xFF) / 255.0f
        var g = ((pixel shr 8) and 0xFF) / 255.0f
        var b = (pixel and 0xFF) / 255.0f

        r = (r - .5f) / .5f
        g = (g - .5f) / .5f
        b = (b - .5f) / .5f

        val row = i / width
        val col = i % width

        data[row * width + col] = r
        data[1 * height * width + row * width + col] = g
        data[2 * height * width + row * width + col] = b
    }

    return data
}

fun resizeImage(image: BufferedImage, width: Int, height: Int): BufferedImage {
    val resized = BufferedImage(width, height, image.type)
    val g = resized.createGraphics()
    g.setRenderingHints(
        mapOf(
            RenderingHints.KEY_INTERPOLATION to RenderingHints.VALUE_INTERPOLATION_BILINEAR,
        )
    )
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    return resized
}
