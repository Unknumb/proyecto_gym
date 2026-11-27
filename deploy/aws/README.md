# Lanzar Sustituto Seguro en AWS Academy (Learner Lab)

Esta guía levanta la página web, la API de cuentas y la base de datos PostgreSQL en el
Learner Lab de AWS Academy, con una sola plantilla de CloudFormation
([`sustituto-seguro.yaml`](sustituto-seguro.yaml)). Toma unos 20 minutos, la mayoría
esperando que AWS cree la base.

## Qué se crea

```
Internet ──HTTPS──► EC2 (Amazon Linux 2023, IP fija)                RDS PostgreSQL (privada)
                    ├─ Caddy: certificado HTTPS gratuito (sslip.io)   └─ base "sustituto"
                    └─ Gunicorn + Flask: página y API /api/* ────5432──►
```

| Recurso | Para qué | Tamaño |
|---|---|---|
| VPC con dos subredes | Red propia; RDS exige dos zonas | — |
| EC2 `t3.micro` + IP elástica | Página, API y HTTPS, siempre en la misma dirección | 1 servidor |
| RDS PostgreSQL `db.t3.micro` | Cuentas, perfiles, rutinas, sesiones, pesos y aportes | 20 GB |
| Grupos de seguridad | Web abierta en 80/443; la base solo acepta al servidor web | — |

La plantilla respeta los límites del Learner Lab: regiones `us-east-1` o `us-west-2`,
instancias chicas, sin Multi-AZ ni monitoreo mejorado y **sin crear roles de IAM**.

## Paso a paso

1. **Inicia el laboratorio.** En AWS Academy, entra al curso → *Modules* → *Learner Lab*
   → **Start Lab**. Cuando el círculo junto a *AWS* se ponga verde, haz clic en **AWS**.
2. **Revisa la región.** Arriba a la derecha de la consola debe decir **N. Virginia**
   (`us-east-1`). Si no, cámbiala.
3. **Crea la pila.** Busca **CloudFormation** → **Create stack** → *With new resources
   (standard)* → *Upload a template file* → elige `deploy/aws/sustituto-seguro.yaml` →
   **Next**.
4. **Completa los parámetros.**
   - *Stack name*: `sustituto-seguro`.
   - **ClaveBaseDatos**: una contraseña de 12 a 40 letras o números, sin símbolos (por
     ejemplo `Gimnasio2026Duoc`). Guárdala: es la del administrador de PostgreSQL.
   - Lo demás se deja como viene: rama `main`, `t3.micro`, HTTPS `si`, llave `vockey`.

   **Next** → **Next** → **Submit**.
5. **Espera** a que el estado pase a `CREATE_COMPLETE`, unos 15 minutos. RDS es lo que
   más tarda.
6. **Abre la página.** En la pestaña **Outputs**, `DireccionPagina` tiene la dirección,
   por ejemplo `https://3.85.12.40.sslip.io`. El servidor termina de instalarse unos 3
   minutos después de que la pila esté lista. Si la página no carga, espera y recarga.
7. **Pruébala.** Entra a *Mi cuenta* → *Crear cuenta*. Lo que guardes queda en la base de
   RDS y lo ves igual desde otro navegador o desde la app de iPhone.
8. **App de iPhone.** En la app, *Mi cuenta* → *Servidor*, escribe la dirección de
   `DireccionPagina` y crea la cuenta o inicia sesión.

## Cada vez que vuelvas al laboratorio

- Al terminar la sesión del Learner Lab, AWS **detiene** el servidor. Al volver: **Start
  Lab** → consola → **EC2** → *Instances* → `sustituto-seguro` → *Instance state* →
  **Start instance**, si no arrancó solo. La IP elástica mantiene la misma dirección, y
  la página y la API arrancan solas.
- Si RDS aparece detenida (*Stopped*), en **RDS** → *Databases* → `sustituto` →
  *Actions* → **Start**.
- El crédito del laboratorio es limitado. Cuando terminen de evaluar, **borra la pila**:
  CloudFormation → `sustituto-seguro` → **Delete**. Eso borra el servidor, la base y la
  IP, con todas las cuentas.

## Actualizar a una versión nueva

Cuando se suban cambios a `main`, conéctate al servidor (EC2 → la instancia → **Connect**
→ *EC2 Instance Connect* → **Connect**) y ejecuta:

```bash
sudo sustituto-actualizar
```

## Si algo falla

| Síntoma | Qué revisar |
|---|---|
| La pila queda en `ROLLBACK_COMPLETE` | Pestaña **Events**: la primera fila en rojo dice qué recurso falló. Lo más común es una contraseña con símbolos o una región distinta de us-east-1/us-west-2. Bórrala y créala de nuevo |
| La página no carga después de 10 minutos | En el servidor: `sudo tail -50 /var/log/cloud-init-output.log` (instalación) y `sudo journalctl -u sustituto -u caddy -n 50` (servicios) |
| El navegador avisa por el certificado | Caddy lo pide a Let's Encrypt la primera vez; puede tardar un par de minutos. Si sigue fallando, `sudo journalctl -u caddy -n 50` |
| Se quiere servir sin HTTPS | En el servidor: `echo ':80 { reverse_proxy 127.0.0.1:8000 }' \| sudo tee /etc/caddy/Caddyfile && sudo systemctl restart caddy` y entra por `http://<IP>`. La app de iPhone exige HTTPS, así que solo sirve para la web |
| «No se pudo conectar con el servidor» | `curl -s localhost:8000/api/salud` en el servidor debe responder `"ok": true`; si no, `sudo journalctl -u sustituto -n 50` |

## Ver la base de datos

Para mostrar las tablas en la presentación, desde el servidor:

```bash
sudo dnf install -y postgresql16 || sudo dnf install -y postgresql15
source <(sudo cat /etc/sustituto.env | sed 's/^/export /')
psql "$DATABASE_URL" -c '\dt'
psql "$DATABASE_URL" -c 'SELECT id, nombre, creado FROM usuarios'
psql "$DATABASE_URL" -c 'SELECT usuario_id, fecha, titulo, esfuerzo, dolor FROM sesiones ORDER BY fecha DESC LIMIT 10'
```

Las contraseñas no se ven: en la tabla `usuarios` solo está su hash (`scrypt`).

## Sin AWS: en el computador

La misma API corre localmente con SQLite, sin instalar nada más que las dependencias del
proyecto:

```bash
python -m api.servidor          # http://localhost:8000
```
