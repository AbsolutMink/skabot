# Tareas propuestas tras la revisión

## 1. Corregir error tipográfico
- **Descripción:** El comentario en `TradeExecutor.execute_trade` utiliza la conjunción "e" antes de "aplicarlos", lo cual es gramaticalmente incorrecto en español y debería ser "y aplicarlos".
- **Ubicación:** `README.md`, línea 810.
- **Tarea propuesta:** Actualizar el comentario para corregir la conjunción y mantener la documentación interna clara y profesional.

## 2. Corregir falla lógica en la evaluación de señales
- **Descripción:** En `ModuloAnalisisSeñales.check_all_signals`, las comprobaciones para MACD, cruces de EMA y Bandas de Bollinger están encadenadas con `elif` después del bloque del Estocástico. Debido a que ese `elif not signal_direction` siempre se evalúa primero cuando no se ha encontrado señal, el resto de indicadores nunca se evalúa, dejando inoperantes varias fuentes de señal.
- **Ubicación:** `README.md`, líneas 326-382.
- **Tarea propuesta:** Reestructurar la lógica para que cada conjunto de condiciones se evalúe con sentencias `if` independientes (p. ej. convertir los `elif not signal_direction` en `if not signal_direction`) o reorganizar el flujo para garantizar que todas las fuentes de señal puedan ejecutarse cuando corresponda.

## 3. Corregir discrepancia en comentario de documentación
- **Descripción:** En `TradingBotV43.validate_symbols`, el comentario indica que, si no se encuentran símbolos preferidos, se buscará "cualquier par de Forex disponible". Sin embargo, la implementación limita la búsqueda a los primeros 20 símbolos y únicamente a aquellos que contienen "USD", por lo que el comentario no describe con precisión el comportamiento real.
- **Ubicación:** `README.md`, líneas 931-938.
- **Tarea propuesta:** Ajustar el comentario para reflejar las restricciones actuales o modificar el código para cumplir con la descripción del comentario.

## 4. Mejorar cobertura de pruebas
- **Descripción:** No existen pruebas automatizadas que verifiquen el comportamiento de `check_all_signals`, especialmente para garantizar que todas las fuentes de señal se evalúen correctamente y que la priorización funcione tras corregir la falla lógica.
- **Ubicación de referencia:** `README.md`, líneas 313-420.
- **Tarea propuesta:** Añadir pruebas unitarias (por ejemplo, con `pytest`) que simulen marcos de datos con escenarios específicos para cada indicador, comprobando que la función detecte la señal adecuada en cada caso y que no quede bloqueada por condiciones anteriores.
