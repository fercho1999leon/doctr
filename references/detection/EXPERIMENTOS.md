# Plan de experimentos: extractor de campos de comprobantes (LW-DETR)

Base de comparación: `lw_detr_s_fields_r3_photo` (768 px, bf16). En test (168 comprobantes), 73.8 % tiene los 4 campos
requeridos correctos. `cuenta_destino` acierta el 86.7 % y es el cuello de botella.

Ninguna cifra de este documento está medida: son hipótesis con su comando. Con 168 documentos, un comprobante vale
0.6 puntos y una diferencia menor de ~3 puntos entre dos runs es ruido. Compara siempre sobre los mismos documentos
(ver "Comparar dos runs").

## Producción: qué está fallando (export de `/system/ocr/stats`, 22–28 sep 2026)

94 corridas de `lw_detr_s_fields_r3_photo` por la API (93 "solo OCR" + 1 reevaluación): 67 éxito (71 %), 27 parciales,
ningún error del servicio. Latencia mediana 0.54 s (p90 1.4 s; las primeras corridas del 23/09, en ráfaga, 1–5 s).

| campo faltante | corridas | % de 94 | en test (168) |
|---|---|---|---|
| Fecha | 15 | 16 % | 5.6 % de fallo |
| Cuenta destino | 10 (7 `missing`, 3 `unresolved`) | 11 % | 13.3 % de fallo |
| Nº comprobante | 4 (3 junto con cuenta destino) | 4 % | 3.6 % |
| Valor | 1 | 1 % | 2.4 % |

- **La fecha es el primer problema en producción y no en test** (16 % frente a 5.6 %): hay algo en los comprobantes
  reales que el set de test no tiene. Se concentra en días concretos (27/09: 7 de 20), lo que apunta a un formato o
  app de banco concreto. Varios formatos habituales no se reconocían: año de 2 dígitos con mes en letras
  (`28-SEP-26`), mes primero (`Sep 28, 2026`), meses en inglés (`28 September 2026`) y `del 2026`. Ya están
  soportados (tests en `tests/`), pero **sin el texto leído no se puede confirmar que sean estos**.
- **Cuenta destino**: 3 de 10 son `unresolved` (se leyó una cuenta que no es de la empresa: dígitos mal leídos u otra
  cuenta) y 7 `missing`. Los 3 casos en que también falta el Nº de comprobante apuntan a una plantilla que el detector
  no conoce, o a una imagen que no es un comprobante.
- El valor casi nunca falla: no es donde invertir.

### E0b. Diagnóstico de las 27 parciales (sin GPU, antes de reentrenar)

El export no trae el texto leído, así que no distingue "el detector no encontró la región" de "encontró el texto pero
no se pudo interpretar". Son dos arreglos distintos (anotar más ejemplos o ampliar el parser):

```bash
# Descarga las imágenes de las corridas parciales (por Hash / ID desde el ERP) a parciales/ y:
python references/detection/kie_inference.py --checkpoint <ruta>/lw_detr_s_fields_r3_photo.pt \
    --top-k-per-class 1 --merge-fragments --fallback --straighten --json parciales.json parciales/*
```

Para cada campo que falta, mira en `parciales.json`:
- **Lista vacía**: el detector no encontró la región. Marca la corrida "Para entrenar" en el ERP, anótala en Document AI
  y súmala al set (E1). Es aprendizaje activo: las corridas parciales son justo los ejemplos que el modelo no sabe resolver.
- **`value` con texto pero `normalized` vacío**: es un problema del parser. Pásame esos textos (solo la fecha o el
  número, no nombres ni cuentas) y amplío `extract_key` con un test por formato.

Para el ERP, que el export incluya por cada campo faltante el `value` leído, el `detection_score` y el `source`: con eso
este diagnóstico sale directo del CSV.

## Reglas para que los runs sean comparables

1. **Misma partición.** No regeneres `doctr/` con otra semilla ni con `--group-by-key` entre runs. Si E0 detecta
   fugas y regeneras, vuelve a evaluar la base sobre la partición nueva.
2. **Mismo postproceso en todos los runs:** `--top-k-per-class 1 --merge-fragments --fallback --straighten`. Sin
   `--min-score`, o con umbrales ajustados en `val` y aplicados en `test`; nunca ajustados en `test`.
3. **Métrica principal:** la tabla *One value per field* y la línea "Documents with every required field right as one
   value" de `evaluate_fields.py`. No depende de cuántas regiones se conserven, al contrario del *key match*.
4. **`test` una sola vez por candidato final.** Las decisiones se toman en `val`.

## Variables comunes (Colab)

```python
DRIVE = "/content/drive/MyDrive/doctr_data"
DATA = "/content/datasets/doctr"
# Copia aquí los argumentos de la base: están en f"{DRIVE}/checkpoints/lw_detr_s_fields_r3_photo.json" -> "args"
BASE = (
    "--pretrained --labels-name labels_layout.json --input_size 768 --device 0 --amp --amp-dtype bfloat16 "
    "-b 8 --epochs 100 --no-hflip --photo-aug --perspective 0.2 "
    "--early-stop --early-stop-epochs 15 --early-stop-delta 0.001 -j 4"
)
POST = "--top-k-per-class 1 --merge-fragments --fallback --straighten"


def train(name, extra=""):
    get_ipython().system(
        f"python references/layout/train.py lw_detr_s {BASE} {extra} --train_path {DATA}/train "
        f"--val_path {DATA}/val --output_dir {DRIVE}/checkpoints --name {name}"
    )


def evaluate(name, split="val"):
    get_ipython().system(f"mkdir -p {DRIVE}/eval")
    get_ipython().system(
        f"python references/detection/evaluate_fields.py --checkpoint {DRIVE}/checkpoints/{name}.pt "
        f"--data {DATA}/{split} {POST} --output {DRIVE}/eval/{name}_{split}"
    )
```

Revisa que `BASE` coincida con el JSON de r3_photo antes de lanzar nada. Si no coincide, el primer experimento compara
también esas diferencias.

## Costo

Tarifas aproximadas de Colab: A100 ≈ 11–13 unidades/h, L4 ≈ 4.5–5 unidades/h, T4 ≈ 2 unidades/h. Consulta
*Recursos* en tu sesión. Estimación a medir: con unas 800 imágenes de entrenamiento, 768 px y batch 8, una época de
`lw_detr_s` debería rondar un minuto en L4, así que 100 épocas son 1.5–2 h (≈ 8–10 unidades). En A100 debería tardar
la mitad de tiempo con un costo parecido o algo mayor. **Mide la primera época** y multiplica: el early stopping suele
cortar antes. Cada evaluación en `val` + `test` son unos minutos (< 1 unidad).

## E0. Base con las métricas nuevas y control de fugas (sin GPU)

Hazlo antes que todo lo demás; se puede correr en el Windows o en el Mac (env `doctr`).

```bash
# 1) ¿Hay comprobantes repetidos entre train y test (captura + foto de la misma transferencia)?
python references/detection/convert_documentai.py --input <carpetas Document AI> --output /tmp/doctr_audit \
    --val-ratio <el de siempre> --test-ratio <el de siempre> --seed <el de siempre>
#    -> busca "[audit] N annotated identifiers appear in more than one split" y "identifiers_shared_across_splits"
#       en /tmp/doctr_audit/audit.json. Usa los mismos argumentos con los que generaste doctr/: la partición sale igual.

# 2) La base con la métrica de un valor por campo
python references/detection/evaluate_fields.py --checkpoint <ruta>/lw_detr_s_fields_r3_photo.pt \
    --data docs/ENTRENAMIENTO/doctr/test --top-k-per-class 1 --merge-fragments --fallback --straighten \
    --output docs/ENTRENAMIENTO/eval_r3_photo_test
```

Qué mirar:
- Si hay identificadores compartidos, el 73.8 % es optimista. Regenera con `--group-by-key`, reevalúa la base y usa
  esa partición en adelante.
- En la tabla *One value per field*, la columna *annotated with letters only* de `cuenta_destino` cuantifica cuánto del
  86.7 % es anotación (se anotó el nombre del titular) y no error del modelo.

Costo: 0 unidades.

## E1. Anotación de `cuenta_destino` (datos, no GPU): la palanca principal

11 de los 23 fallos de cuenta son comprobantes donde solo se anotó el nombre. Mientras las plantillas de banco no se
anoten igual, el modelo aprende dos cosas contradictorias para una misma clase.

Opción A (recomendada): en Document AI, anotar en **todos** los comprobantes el número enmascarado como
`cuenta_destino`, y el nombre del titular, si se quiere, en una clase nueva `titular_destino`.
Opción B: anotar el nombre y el número como dos entidades `cuenta_destino` separadas (el converter ya admite varias
cajas por clase).

Lista de documentos a corregir (desde el JSON de E0):

```python
import json

r = json.load(open("docs/ENTRENAMIENTO/eval_r3_photo_test.json"))
print([d["id"] for d in r["per_document"] if d["fields"]["cuenta_destino"]["annotation_kind"] == "letters"])
```

Haz lo mismo sobre `train` y `val`: el problema está sobre todo en el entrenamiento. Después: reconvertir, entrenar
con la receta de la base (`train("r4_cuenta")`) y evaluar. Costo: ≈ 8–10 unidades (L4).

## E2. Learning rate de fine-tuning

`layout/train.py` usa por defecto `--lr 1e-3 --backbone-lr 1e-3`. La receta de fine-tuning de LW-DETR usa ~1e-4 para
el decoder y algo más para el encoder. Con un modelo preentrenado y ~800 imágenes, 1e-3 puede estar borrando el
preentrenamiento.

```python
train("e2_lr1e4", "--lr 1e-4 --backbone-lr 1.5e-4")
evaluate("e2_lr1e4")
```

Costo: ≈ 8–10 unidades (L4).

## E3. EMA de pesos

`--ema` evalúa, selecciona y guarda un promedio exponencial de los pesos (decay 0.993, rampa de 100 iteraciones),
como en LW-DETR/RT-DETR. En datasets pequeños suele estabilizar la época elegida.

```python
train("e3_ema", "--ema")
evaluate("e3_ema")
```

Costo: ≈ 8–10 unidades (L4); el EMA agrega una copia del modelo en memoria y es despreciable en tiempo.

## E4. Selección del checkpoint por mAP

Hoy se guarda el checkpoint de menor pérdida de validación. En DETR la pérdida mezcla términos que dependen del
matching y sigue a la precisión de forma floja. `--select-by map` guarda el de mejor mAP@[.5:.95] y aplica el early
stopping sobre esa misma cantidad.

```python
train("e4_map", "--select-by map --save-interval-epoch")
evaluate("e4_map")
```

Con `--save-interval-epoch` puedes además evaluar 3–4 épocas alrededor del mejor mAP con `evaluate_fields.py` y ver
si la métrica de campo sigue al mAP: `evaluate("e4_map_epoch37")`. Costo: ≈ 8–10 unidades (L4) + evaluaciones.

## E5. Combinación

Solo con lo que ganó en `val` en E2–E4, por ejemplo:

```python
train("e5_combo", "--lr 1e-4 --backbone-lr 1.5e-4 --ema --select-by map")
evaluate("e5_combo")
evaluate("e5_combo", "test")
```

Costo: ≈ 8–10 unidades (L4).

## E6. Resolución de entrada 1024

Los dígitos de la cuenta enmascarada son pequeños en fotos. Solo vale la pena si los fallos restantes de
`cuenta_destino` son de localización (baja IoU o regiones partidas en el reporte) y no de anotación.

```python
train("e6_1024", "--input_size 1024 -b 4")  # batch 4 por memoria en L4; en A100 puedes dejar 8
evaluate("e6_1024")
```

Costo: ≈ 1.8× el tiempo por época, ≈ 15–18 unidades (L4). En producción la inferencia también es ~1.8× más lenta.

## E7. Varianza (opcional, para decidir entre dos candidatos cercanos)

Con 5 folds agrupados se mide cuánto cambia la métrica solo por la partición:

```bash
for f in 0 1 2 3 4; do python references/detection/convert_documentai.py --input <...> \
    --output /content/datasets/fold$f --folds 5 --fold $f --group-by-key; done
```

Entrena la receta ganadora en cada fold. Costo: 5 × 8–10 ≈ 40–50 unidades (L4). Solo si E2–E5 dejan dos candidatos a
menos de 3 puntos.

## Comparar dos runs sobre los mismos documentos

```python
import json

a = {d["id"]: d["all_required_value_ok"] for d in json.load(open("eval/base_test.json"))["per_document"]}
b = {d["id"]: d["all_required_value_ok"] for d in json.load(open("eval/e5_combo_test.json"))["per_document"]}
only_a = [k for k in a if a[k] and not b[k]]
only_b = [k for k in a if b[k] and not a[k]]
print(
    f"base {sum(a.values())}  candidato {sum(b.values())}  gana solo base {len(only_a)}  gana solo candidato {len(only_b)}"
)
# Prueba de McNemar (exacta): diferencia real si p < 0.05
from scipy.stats import binomtest

n = len(only_a) + len(only_b)
print("p =", binomtest(len(only_b), n, 0.5).pvalue if n else 1.0)
```

## Orden sugerido y presupuesto

| paso | qué | GPU | unidades aprox. |
|---|---|---|---|
| E0 | fugas + métrica nueva sobre la base | no | 0 |
| E0b | diagnóstico de las 27 parciales de producción | no | 0 |
| E1 | reanotar `cuenta_destino` y reentrenar | L4 | 8–10 |
| E2–E4 | LR, EMA, selección por mAP (en paralelo si tienes varias sesiones) | L4 | 25–30 |
| E5 | combinación ganadora + test | L4 | 8–10 |
| E6 | 1024 px (condicional) | L4 | 15–18 |
| E7 | 5 folds (opcional) | L4 | 40–50 |

Sin E6 ni E7: unas 45–50 unidades.
