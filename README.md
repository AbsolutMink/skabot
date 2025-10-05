# Skabot Trading Bot v4.3

## Configuración de credenciales

El bot obtiene las credenciales sensibles desde variables de entorno. Puedes
establecerlas directamente en el sistema o definir un archivo `.env` en la raíz
del proyecto (también puedes indicar otra ruta mediante la variable
`SKABOT_ENV_FILE`). Durante el arranque se cargarán automáticamente y se
validará que todas las claves obligatorias estén disponibles.

### Variables requeridas

| Variable | Descripción |
| --- | --- |
| `MT5_LOGIN` | Número de cuenta de MetaTrader 5. |
| `MT5_PASSWORD` | Contraseña de la cuenta de MetaTrader 5. |
| `MT5_SERVER` | Servidor al que debe conectarse el bot. |
| `TELEGRAM_TOKEN` | Token del bot de Telegram. |
| `TELEGRAM_CHAT_ID` | Identificador del chat o canal de Telegram donde enviar las notificaciones. |

### Variables opcionales

| Variable | Descripción | Valor por defecto |
| --- | --- | --- |
| `MT5_TIMEOUT` | Tiempo de espera para las operaciones con MT5 (ms). | `60000` |
| `MT5_MAGIC_NUMBER` | Identificador mágico para las órdenes generadas. | `234567` |
| `SKABOT_ENV_FILE` | Ruta personalizada para el archivo `.env`. | `.env` |

### Ejemplo de `.env`

```
MT5_LOGIN=12345678
MT5_PASSWORD=contraseña_segura
MT5_SERVER=Deriv-Demo
MT5_TIMEOUT=60000
MT5_MAGIC_NUMBER=234567
TELEGRAM_TOKEN=1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ
TELEGRAM_CHAT_ID=123456789
```

Si falta alguna de las credenciales requeridas el bot mostrará un mensaje de
error y se detendrá para evitar iniciar con datos incompletos.
