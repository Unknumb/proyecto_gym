// Casos de prueba de la app de iPhone: corre el generador de rutinas de la página
// (web/app.html) sobre perfiles fijos y guarda lo que arma, con los ejercicios por
// ejercicio_id. Los tests de Swift (RutinasTests) comprueban que la app arme lo mismo.
//
//   node scripts/casos_rutina_ios.mjs
//
// Lee el generador directamente de web/app.html, sin copiarlo: si cambia la web y la
// app no, el test de Swift falla.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';

const raiz = join(dirname(fileURLToPath(import.meta.url)), '..');
const html = readFileSync(join(raiz, 'docs/index.html'), 'utf8');
const D = JSON.parse(html.match(/<script type="application\/json" id="datos">([\s\S]*?)<\/script>/)[1].replace(/<\\\//g, '</'));
const fuente = readFileSync(join(raiz, 'web/app.html'), 'utf8');

const tramo = (desde, hasta) => {
  const a = fuente.indexOf(desde), b = fuente.indexOf(hasta, a);
  if (a < 0 || b < 0) throw new Error(`No se encontró el tramo ${desde} … ${hasta}`);
  return fuente.slice(a, b);
};
const codigo = [
  "const A = Object.fromEntries(D.articulaciones.map((a, k) => [a, k]));",
  "const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;",
  "const tiene = (obj, k) => typeof k === 'string' && Object.prototype.hasOwnProperty.call(obj, k);",
  tramo('  // Lesiones tal como las entiende una persona', '  // ═══════════════ Ejercicios conocidos, en español'),
  tramo('  const CURADOS = [', '  const nombreEs = '),
  tramo('  const GRUPO_MUSC = {', '  // ═══════════════ Almacenamiento local'),
  tramo('  const DEPORTES = {', '  // ═══════════════ RUTINA: vista'),
  "globalThis.generarRutina = generarRutina;"
].join('\n');
const contexto = { D, console, hoyISO: () => '2026-11-27' };
vm.createContext(contexto);
vm.runInContext(codigo.replace(/const hoyISO[^\n]*\n/, ''), contexto);

// Perfiles: combinaciones fijas que recorren edades, pesos, deportes, niveles, tipos, días, equipos y lesiones
const deportes = ['ninguno', 'futbol', 'running', 'ciclismo', 'natacion', 'basquet', 'tenis', 'combate', 'escalada', 'otro'];
const niveles = ['principiante', 'intermedio', 'avanzado'];
const objetivos = ['salud', 'fuerza', 'musculo', 'resistencia'];
const tipos = ['auto', 'full', 'tp', 'epp', 'arnold', 'weider'];
const equipos = [[], ['pesas'], ['gym'], ['pesas', 'gym'], ['banda'], ['balon', 'banda'], ['pesas', 'gym', 'banda', 'balon']];
const lesiones = [[], ['rodilla'], ['hombro'], ['lumbar'], ['codo', 'muneca'], ['cadera', 'tobillo'], ['cuello'], ['rodilla', 'hombro', 'lumbar']];
const edades = [12, 16, 25, 34, 45, 62, 75];
const casos = [];
for (let k = 0; k < 60; k++) {
  const perfil = {
    edad: edades[k % edades.length], peso: [60, 72, 115, ''][k % 4], altura: [170, '', 182][k % 3],
    deporte: deportes[k % deportes.length], nivel: niveles[(k * 7) % 3], objetivo: objetivos[(k * 5) % 4],
    dias: 2 + (k % 5), minutos: [30, 45, 60, 75][(k * 3) % 4], tipo: tipos[k % tipos.length],
    equipo: equipos[(k * 3) % equipos.length], lesiones: lesiones[(k * 5) % lesiones.length]
  };
  const r = contexto.generarRutina(perfil);
  casos.push({ perfil, rutina: {
    objetivo: r.objetivo, tipo: r.tipo, porqueTipo: r.porqueTipo, avisos: r.avisos, huecos: r.huecos,
    dias: r.dias.map(d => ({ titulo: d.titulo, equilibrio: !!d.equilibrio, ejercicios: d.ejercicios.map(e => ({
      id: D.ids[e.i], grupo: e.grupo, series: e.series, reps: e.reps, descanso: e.descanso, iso: e.iso, extra: e.extra,
      alternativas: e.alternativas.map(a => D.ids[a]) })) })) } });
}
const salida = join(raiz, 'ios/SustitutoSeguroCore/Tests/SustitutoSeguroCoreTests/Recursos/casos_rutina.json');
writeFileSync(salida, JSON.stringify({ casos }, null, 1));
console.log(`${casos.length} rutinas de la web → ${salida.slice(raiz.length + 1)}`);
