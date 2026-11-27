import Charts
import SwiftUI
import SustitutoSeguroCore

/// Progreso: constancia por semana, dolor por zona, carga por ejercicio, peso corporal e
/// historial. Los mismos cálculos de la página (`Progreso` en SustitutoSeguroCore).
struct ProgresoView: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(Almacen.self) private var almacen
    @State private var ejercicio: String?
    @State private var pesoHoy: Double?
    @State private var borrando: Sesion?

    var body: some View {
        let c = almacen.contenido
        let sesiones = c.entrenamientos
        let hoy = Fechas.iso()
        let r = Progreso.resumen(sesiones, hoy: hoy, objetivo: c.perfil?.dias ?? 3)
        List {
            Section {
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                    Kpi(valor: "\(r.estaSemana)/\(r.objetivo)", texto: "Sesiones esta semana")
                    Kpi(valor: "\(r.racha)", texto: "Semanas seguidas")
                    Kpi(valor: "\(r.total)", texto: "Sesiones en total")
                    Kpi(valor: Progreso.esfuerzoPromedio(sesiones).map { $0.formatted(.number.precision(.fractionLength(1))) } ?? "—",
                        texto: "Esfuerzo promedio (1–10)")
                }
                .padding(.vertical, 4)
                ForEach(Array(Progreso.alertas(sesiones).enumerated()), id: \.offset) { _, a in AvisoView(aviso: a) }
            }

            if sesiones.isEmpty {
                Section {
                    ContentUnavailableView("Aún no hay sesiones", systemImage: "chart.bar",
                                           description: Text("Cuando termines tu primer entrenamiento aquí verás tu constancia, tus cargas y cómo evoluciona el dolor."))
                }
            } else {
                constancia(sesiones, hoy: hoy, meta: r.objetivo)
                dolor(sesiones)
                cargas(sesiones)
            }
            pesoCorporal(c.pesos)
            if !sesiones.isEmpty { historial(sesiones) }
        }
        .navigationTitle("Progreso")
        .confirmationDialog("¿Borrar esta sesión?", isPresented: Binding(get: { borrando != nil }, set: { if !$0 { borrando = nil } }),
                            titleVisibility: .visible, presenting: borrando) { s in
            Button("Borrar", role: .destructive) { almacen.contenido.entrenamientos.removeAll { $0.uid == s.uid } }
        }
    }

    // MARK: Secciones

    private static func etiqueta(_ fechas: [String], _ k: Int?) -> String {
        guard let k, fechas.indices.contains(k) else { return "" }
        return Fechas.corta(fechas[k])
    }

    private func constancia(_ sesiones: [Sesion], hoy: String, meta: Int) -> some View {
        let semanas = Progreso.porSemana(sesiones, hoy: hoy)
        return Section {
            Chart {
                ForEach(Array(semanas.enumerated()), id: \.offset) { k, w in
                    BarMark(x: .value("Semana", Fechas.corta(w.lunes)), y: .value("Sesiones", w.sesiones))
                        .foregroundStyle(k == semanas.count - 1 ? Color.accentColor : Color.accentColor.opacity(0.5))
                        .cornerRadius(3)
                }
                RuleMark(y: .value("Meta", meta))
                    .foregroundStyle(.orange)
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                    .annotation(position: .top, alignment: .leading) { Text("Meta: \(meta)").font(.caption2).foregroundStyle(.orange) }
            }
            .chartYScale(domain: 0...max(meta, semanas.map(\.sesiones).max() ?? 0, 1))
            .frame(height: 180)
        } header: {
            Text("Constancia")
        } footer: {
            Text("Sesiones por semana, últimas 8 semanas (la fecha es el lunes).")
        }
    }

    @ViewBuilder
    private func dolor(_ sesiones: [Sesion]) -> some View {
        let zonas = Progreso.zonasConDolor(sesiones)
        let conDolor = Array(sesiones.filter { !$0.dolor.isEmpty }.suffix(12))
        if !zonas.isEmpty && !conDolor.isEmpty {
            Section {
                Chart {
                    ForEach(zonas, id: \.self) { zona in
                        ForEach(Array(conDolor.enumerated()), id: \.offset) { k, s in
                            if let v = s.dolor[zona] {
                                LineMark(x: .value("Sesión", k), y: .value("Dolor", v))
                                    .foregroundStyle(by: .value("Zona", TablasRutina.nombreLesion(zona)))
                                PointMark(x: .value("Sesión", k), y: .value("Dolor", v))
                                    .foregroundStyle(by: .value("Zona", TablasRutina.nombreLesion(zona)))
                            }
                        }
                    }
                    RuleMark(y: .value("Ideal", 3)).foregroundStyle(.gray.opacity(0.4)).lineStyle(StrokeStyle(dash: [3, 3]))
                }
                .chartYScale(domain: 0...10)
                .chartXAxis {
                    AxisMarks(values: Array(conDolor.indices)) { v in
                        AxisValueLabel { Text(Self.etiqueta(conDolor.map(\.fecha), v.as(Int.self))) }
                    }
                }
                .frame(height: 190)
            } header: {
                Text("Dolor por zona")
            } footer: {
                Text("Escala 0 a 10, últimas sesiones. Lo ideal es que se mantenga bajo 3.")
            }
        }
    }

    @ViewBuilder
    private func cargas(_ sesiones: [Sesion]) -> some View {
        let ids = Progreso.ejerciciosConCarga(sesiones)
        if let elegido = ejercicio.flatMap({ ids.contains($0) ? $0 : nil }) ?? ids.first {
            let puntos = Progreso.cargas(sesiones, id: elegido)
            Section {
                Picker("Ejercicio", selection: Binding(get: { elegido }, set: { ejercicio = $0 })) {
                    ForEach(ids, id: \.self) { Text(catalogo.nombre($0)).tag($0) }
                }
                Chart(Array(puntos.enumerated()), id: \.offset) { k, p in
                    LineMark(x: .value("Sesión", k), y: .value("Kg", p.kg))
                    PointMark(x: .value("Sesión", k), y: .value("Kg", p.kg))
                }
                .chartXAxis {
                    AxisMarks(values: Array(puntos.indices)) { v in
                        AxisValueLabel { Text(Self.etiqueta(puntos.map(\.fecha), v.as(Int.self))) }
                    }
                }
                .chartYScale(domain: .automatic(includesZero: false))
                .frame(height: 180)
            } header: {
                Text("Carga por ejercicio")
            } footer: {
                if let a = puntos.first, let b = puntos.last, puntos.count > 1 {
                    let cambio = b.kg - a.kg
                    Text("Kilos de la serie más pesada en cada sesión. Cambio: \(cambio >= 0 ? "+" : "")\(cambio.formatted(.number.precision(.fractionLength(0...1)))) kg.")
                } else {
                    Text("Kilos de la serie más pesada en cada sesión.")
                }
            }
        }
    }

    private func pesoCorporal(_ pesos: [Pesaje]) -> some View {
        let ultimos = Array(pesos.suffix(15))
        return Section {
            HStack {
                TextField("Registrar hoy (kg)", value: $pesoHoy, format: .number).tecladoDecimal()
                Button("Guardar") {
                    if let kg = pesoHoy, (30...250).contains(kg) { almacen.registrarPeso(kg); pesoHoy = nil }
                }
                .disabled(!(30...250).contains(pesoHoy ?? 0))
            }
            if ultimos.count > 1 {
                Chart(ultimos) { p in
                    LineMark(x: .value("Fecha", Fechas.corta(p.fecha)), y: .value("Kg", p.kg)).foregroundStyle(.orange)
                    PointMark(x: .value("Fecha", Fechas.corta(p.fecha)), y: .value("Kg", p.kg)).foregroundStyle(.orange)
                }
                .chartYScale(domain: .automatic(includesZero: false))
                .frame(height: 170)
            }
        } header: {
            Text("Peso corporal")
        } footer: {
            Text(pesos.isEmpty ? "Opcional. Útil si quieres seguir tu peso junto con tu entrenamiento."
                 : pesos.count == 1 ? "Registra otro día para ver la evolución." : "Últimos \(ultimos.count) registros.")
        }
    }

    private func historial(_ sesiones: [Sesion]) -> some View {
        Section("Historial") {
            ForEach(Array(sesiones.reversed().prefix(30))) { s in
                VStack(alignment: .leading, spacing: 3) {
                    Text("\(Textos.capitalizar(Fechas.larga(s.fecha))) · \(s.titulo)").font(.subheadline.bold())
                    Text("\(s.ejercicios.count) ejercicios · \(s.series) series · \(s.minutos) min · esfuerzo \(s.esfuerzo)/10"
                         + (s.dolor.isEmpty ? "" : " · dolor " + s.dolor.sorted { $0.key < $1.key }.map { "\(TablasRutina.nombreLesion($0.key).lowercased()) \($0.value)" }.joined(separator: ", ")))
                        .font(.caption).foregroundStyle(.secondary)
                    if !s.notas.isEmpty { Text(s.notas).font(.caption).italic() }
                }
                .swipeActions { Button("Borrar", role: .destructive) { borrando = s } }
            }
        }
    }
}

private struct Kpi: View {
    let valor: String
    let texto: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(valor).font(.title2.bold()).monospacedDigit()
            Text(texto).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
