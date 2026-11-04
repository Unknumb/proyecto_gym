// swift-tools-version: 5.9
// Lógica de la app Sustituto Seguro: inferencia del perfil de exigencia,
// recomendaciones precalculadas y exportación de datos. Sin dependencias, para
// probarla en el Mac con `swift test` y usarla desde la app de iPhone.
import PackageDescription

let package = Package(
    name: "SustitutoSeguroCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "SustitutoSeguroCore", targets: ["SustitutoSeguroCore"])],
    targets: [
        .target(name: "SustitutoSeguroCore"),
        // `swift run verificar`: compara con scikit-learn sin Xcode (las Command Line
        // Tools no traen XCTest ni los macros de Swift Testing).
        .executableTarget(name: "verificar", dependencies: ["SustitutoSeguroCore"]),
        .testTarget(name: "SustitutoSeguroCoreTests", dependencies: ["SustitutoSeguroCore"],
                    resources: [.copy("Recursos")]),
    ]
)
