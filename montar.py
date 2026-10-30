"""Gera o notebook independente. Execute: python3 montar.py."""

import json
import csv
import base64
import textwrap
from pathlib import Path


RAIZ = Path(__file__).resolve().parent
celulas = []


def celula(tipo, texto):
    item = {'cell_type': tipo, 'id': f'celula-{len(celulas):02d}', 'metadata': {},
            'source': textwrap.dedent(texto).strip() + '\n'}
    if tipo == 'code':
        item.update(execution_count=None, outputs=[])
    celulas.append(item)


def md(texto):
    celula('markdown', texto)


def codigo(texto):
    celula('code', texto)


md('''
# Detecção de EPIs em obras: YOLOv8n versus YOLO11n

**Objetivo:** treinar duas versões do YOLO para detectar equipamentos de proteção
individual (EPIs), pessoas e as demais classes da base Construction-PPE.
Comparar as métricas no conjunto de teste e apresentar imagens com as previsões
de bounding boxes, classes e confiança.

**Como reproduzir:** selecione uma GPU em **Ambiente de execução > Alterar tipo de
ambiente de execução** e clique em **Executar tudo**. Ao terminar, salve este
notebook com as novas saídas e compartilhe o link com permissão de leitura.

Este notebook baixa a base e os pesos pré-treinados oficiais. Usa a API Python
da Ultralytics indicada no enunciado e faz fine-tuning dos dois modelos.
Quando os arquivos de uma execução já realizada estão presentes no projeto,
o gerador inclui seus resultados no início deste notebook para consulta.
As células de código seguintes permitem reproduzir o experimento completo.
''')

md('''
## 1. Problema, base e protocolo

Detecção de objetos responde **o que aparece e onde aparece**. Cada previsão
contém a classe, uma caixa e um valor de confiança.

A [Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/)
possui, conforme a documentação, 1.416 imagens: 1.132 de treino, 143 de validação
e 141 de teste. Contém 11 classes: `helmet`, `gloves`, `vest`, `boots`, `goggles`,
`none`, `Person`, `no_helmet`, `no_goggle`, `no_gloves` e `no_boots`.
As classes e seus nomes originais são preservados, inclusive a classe genérica `none`.
Há classes explícitas para ausência de alguns EPIs; ausência de uma detecção
de equipamento, isoladamente, não comprova que ele esteja ausente.

**Protocolo:**

- YOLOv8n e YOLO11n iniciam com pesos oficiais pré-treinados em COCO.
- Mesmos dados, 30 épocas, imagens de 640 pixels, batch 16 e seed 42.
- Mesmo otimizador AdamW e hiperparâmetros explícitos; sem parada antecipada.
- O treino aprende os pesos. A validação escolhe o checkpoint `best.pt`.
- O teste fica reservado para avaliar os checkpoints já escolhidos.
- Uma execução por modelo. A variante `n` aproxima o porte dos modelos,
  mas não torna suas arquiteturas ou quantidades de parâmetros idênticas.

Antes do treino, o notebook verifica as anotações e elimina duplicatas exatas
por conteúdo na ordem treino, validação e teste, preservando os arquivos originais.
Os totais efetivamente usados e as exclusões serão exibidos abaixo.
''')

md('''
## 2. Instalação

A versão da Ultralytics está fixada para manter a mesma API nas duas execuções.
O PyTorch fornecido pelo Colab é preservado. A lista completa de versões será exportada.
''')
dependencias = RAIZ.joinpath('requirements.txt').read_text(encoding='utf-8').splitlines()
codigo('import subprocess\nimport sys\n\n'
       f'subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *{dependencias!r}])')

md('''
## 3. Configuração e ambiente

Para encurtar a execução, altere `EPOCHS` para 15 **antes** de treinar os dois
modelos. Se faltar memória, altere `BATCH` para 8 antes de ambos os treinos.
Os valores usados são registrados automaticamente.
''')
codigo('''
import os
import json
import random
import time
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

BASE = (Path.cwd() / 'trabalho').resolve()
BASE.mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(BASE / 'config')
os.environ['WANDB_MODE'] = 'disabled'

import torch
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import ultralytics
from PIL import Image, ImageDraw
from IPython.display import display, Markdown, FileLink
from ultralytics import YOLO, settings

EPOCHS = 30
BATCH = 16
IMGSZ = 640
SEED = 42
CONF_VISUAL = 0.25
MODELOS = {'YOLOv8n': 'yolov8n.pt', 'YOLO11n': 'yolo11n.pt'}
if not torch.cuda.is_available():
    raise RuntimeError('GPU indisponível. No Colab, selecione GPU em Ambiente de execução > Alterar tipo de ambiente de execução e execute novamente.')
DEVICE = 0
PASTA = BASE / 'execucoes' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
PASTA.mkdir(parents=True)
settings.update({'datasets_dir': str(BASE / 'dados'), 'sync': False, 'wandb': False})

PARAMETROS = dict(epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=SEED,
                 optimizer='AdamW', lr0=0.001, lrf=0.01, weight_decay=0.0005,
                 patience=0, deterministic=True, device=DEVICE, workers=2,
                 amp=True, cache=False, close_mosaic=5, plots=True, save=True,
                 save_period=10, fraction=1.0)
AMBIENTE = {'python': sys.version, 'plataforma': platform.platform(),
            'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'inicio_utc': datetime.now(timezone.utc).isoformat()}
(PASTA / 'configuracao.json').write_text(json.dumps(
    {'modelos': MODELOS, 'treino': PARAMETROS, 'ambiente': AMBIENTE,
     'confianca_visual': CONF_VISUAL}, indent=2, ensure_ascii=False), encoding='utf-8')
(PASTA / 'dependencias.txt').write_text(
    subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
print(json.dumps(AMBIENTE, indent=2, ensure_ascii=False))
print('Saída:', PASTA)
CHECKPOINTS, TEMPOS = {}, {}
''')

md('''
## 4. Download e auditoria

A configuração oficial faz o download automático da base. A auditoria usa
SHA-256 dos arquivos para evitar que uma cópia exata seja usada em divisões
diferentes. Isso não detecta imagens parecidas, recortes ou recompressões.
''')
codigo(RAIZ.joinpath('auditoria.py').read_text(encoding='utf-8'))
codigo('''
from ultralytics.data.utils import check_det_dataset

dados = check_det_dataset('construction-ppe.yaml')
NOMES = {int(k): v for k, v in dados['names'].items()}
LISTAS, RESUMO, DUPLICATAS, CONTAGENS = preparar_divisoes(
    {s: dados[s] for s in ['train', 'val', 'test']}, len(NOMES))

# Novos manifestos apontam para os arquivos originais, sem movê-los.
config_base = {'path': str(Path(dados['path']).resolve()), 'names': NOMES}
for split, imagens in LISTAS.items():
    manifesto = PASTA / f'{split}.txt'
    manifesto.write_text(''.join(str(p) + '\\n' for p in imagens), encoding='utf-8')
    config_base[split] = str(manifesto)
DATA_YAML = PASTA / 'base.yaml'
DATA_YAML.write_text(yaml.safe_dump(config_base, sort_keys=False), encoding='utf-8')
auditoria = {'resumo': RESUMO, 'duplicatas': DUPLICATAS, 'boxes_por_classe': CONTAGENS}
(PASTA / 'auditoria.json').write_text(json.dumps(auditoria, indent=2, ensure_ascii=False), encoding='utf-8')

# Seleção das imagens visuais feita antes de conhecer os resultados.
AMOSTRAS = random.Random(SEED).sample(LISTAS['test'], min(6, len(LISTAS['test'])))
(PASTA / 'amostras.json').write_text(json.dumps([p.name for p in AMOSTRAS], indent=2), encoding='utf-8')
display(pd.DataFrame(RESUMO).T)
distribuicao = pd.DataFrame(CONTAGENS, index=[NOMES[i] for i in range(len(NOMES))])
distribuicao.index.name = 'classe'
distribuicao.to_csv(PASTA / 'distribuicao.csv')
display(distribuicao)
ax = distribuicao.plot.bar(figsize=(12, 4), title='Quantidade de boxes por classe e divisão')
ax.set_ylabel('Boxes anotadas')
plt.tight_layout()
plt.savefig(PASTA / 'distribuicao.png', dpi=140)
plt.show()
print('Duplicatas excluídas:', len(DUPLICATAS))
''')

md('''
## 5. Exemplos das anotações

Estas caixas são as anotações de referência, usadas no aprendizado.
As previsões dos modelos serão mostradas depois, em imagens de teste.
''')
codigo('''
def imagem_anotada(caminho):
    imagem = Image.open(caminho).convert('RGB')
    desenho = ImageDraw.Draw(imagem)
    w, h = imagem.size
    for classe, x, y, largura, altura in ler_boxes(arquivo_label(caminho), len(NOMES)):
        canto = ((x - largura / 2) * w, (y - altura / 2) * h,
                 (x + largura / 2) * w, (y + altura / 2) * h)
        cor = tuple(int(c * 255) for c in plt.get_cmap('tab20')(classe)[:3])
        desenho.rectangle(canto, outline=cor, width=max(2, w // 350))
        desenho.text((max(0, canto[0]), max(0, canto[1] - 12)), NOMES[classe], fill='white', stroke_width=1, stroke_fill='black')
    return imagem

exemplos = random.Random(SEED).sample(LISTAS['train'], min(3, len(LISTAS['train'])))
fig, eixos = plt.subplots(1, len(exemplos), figsize=(16, 5), squeeze=False)
for eixo, caminho in zip(eixos[0], exemplos):
    eixo.imshow(imagem_anotada(caminho))
    eixo.set_title(caminho.name[:45])
    eixo.axis('off')
plt.tight_layout()
plt.savefig(PASTA / 'anotacoes_treino.jpg', dpi=130)
plt.show()
''')

md('''
## 6. Treinamento do YOLOv8n

O modelo recebe as imagens e as caixas de referência da base. A cada época,
a biblioteca registra perdas e métricas de validação. `best.pt` é escolhido
automaticamente pela função de fitness de validação da Ultralytics.
''')
codigo('''
inicio = time.perf_counter()
modelo = YOLO(MODELOS['YOLOv8n'])
modelo.train(data=str(DATA_YAML), project=str(PASTA / 'treinos'), name='yolov8n', **PARAMETROS)
CHECKPOINTS['YOLOv8n'] = Path(modelo.trainer.best)
TEMPOS['YOLOv8n'] = (time.perf_counter() - inicio) / 60
del modelo
torch.cuda.empty_cache()
print('Checkpoint:', CHECKPOINTS['YOLOv8n'])
''')

md('''
## 7. Treinamento do YOLO11n

Usa os mesmos manifestos e parâmetros. O histórico de cada modelo fica em
uma pasta própria para evitar que os resultados sejam sobrescritos.
''')
codigo('''
inicio = time.perf_counter()
modelo = YOLO(MODELOS['YOLO11n'])
modelo.train(data=str(DATA_YAML), project=str(PASTA / 'treinos'), name='yolo11n', **PARAMETROS)
CHECKPOINTS['YOLO11n'] = Path(modelo.trainer.best)
TEMPOS['YOLO11n'] = (time.perf_counter() - inicio) / 60
del modelo
torch.cuda.empty_cache()
print('Checkpoint:', CHECKPOINTS['YOLO11n'])
''')

md('''
## 8. Avaliação no teste e comparação quantitativa

- **Precisão:** proporção das detecções consideradas positivas que estão corretas.
- **Recall:** proporção dos objetos anotados que foram encontrados.
- **IoU:** interseção dividida pela união entre caixa prevista e caixa de referência.
- **mAP50:** média da precisão média por classe, exigindo IoU de pelo menos 0,50.
- **mAP50-95:** média em limiares de IoU de 0,50 até 0,95; exige localização mais precisa.

As métricas vão de 0 a 1. Quanto maior, melhor. Precisão e recall são os resumos
calculados pela Ultralytics no ponto de máximo F1 de suas curvas; não são
necessariamente os valores do limiar de 0,25 usado nas imagens demonstrativas.
O mAP considera a curva de confiança, usando `conf=0.001` para avaliação.

A velocidade abaixo é o tempo da etapa de inferência por imagem informado pelo
validador, com batch 1, mesma GPU e precisão FP32. Exclui pré e pós-processamento
e é uma medição aproximada desta execução. O tempo de treino inclui carregamento,
validação e operações de saída, além da otimização dos pesos.
''')
codigo('''
linhas, por_classe = [], []
for nome, checkpoint in CHECKPOINTS.items():
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint ausente: {checkpoint}')
    modelo = YOLO(str(checkpoint))
    metricas = modelo.val(data=str(DATA_YAML), split='test', imgsz=IMGSZ,
                          batch=1, device=DEVICE, half=False, conf=0.001, iou=0.7,
                          plots=True, save_txt=True, save_conf=True,
                          project=str(PASTA / 'teste'), name=nome.lower(), workers=2)
    linhas.append({'modelo': nome, 'precisao': float(metricas.box.mp),
                   'recall': float(metricas.box.mr), 'mAP50': float(metricas.box.map50),
                   'mAP50-95': float(metricas.box.map),
                   'inferencia_ms': float(metricas.speed['inference']),
                   'tempo_treino_min': TEMPOS[nome],
                   'parametros_milhoes': sum(p.numel() for p in modelo.model.parameters()) / 1e6,
                   'pesos_MB': checkpoint.stat().st_size / 1e6})
    # Os arrays seguem ap_class_index, não todas as classes da base.
    por_classe.extend(resumir_classes(nome, NOMES, CONTAGENS['test'],
                                     metricas.box.ap_class_index, metricas.box.p,
                                     metricas.box.r, metricas.box.ap50, metricas.box.ap))
    del modelo
    torch.cuda.empty_cache()

comparacao = pd.DataFrame(linhas)
classes = pd.DataFrame(por_classe)
comparacao.to_csv(PASTA / 'comparacao.csv', index=False)
classes.to_csv(PASTA / 'comparacao_por_classe.csv', index=False)
display(comparacao.round(4))
display(classes.round(4))
fig, eixos = plt.subplots(1, 2, figsize=(13, 4))
comparacao.set_index('modelo')[['precisao', 'recall', 'mAP50', 'mAP50-95']].plot.bar(ax=eixos[0], ylim=(0, 1), rot=0)
eixos[0].set_title('Métricas no teste')
comparacao.set_index('modelo')['inferencia_ms'].plot.bar(ax=eixos[1], rot=0, color=['#345995', '#32936f'])
eixos[1].set_title('Inferência no teste, batch 1 (menor é melhor)')
eixos[1].set_ylabel('ms por imagem, sem pré/pós-processamento')
plt.tight_layout()
plt.savefig(PASTA / 'comparacao.png', dpi=160)
plt.show()
classes.pivot(index='classe', columns='modelo', values='AP50-95').plot.bar(figsize=(13, 4), ylim=(0, 1), rot=40)
plt.title('AP50-95 por classe presente no teste')
plt.tight_layout()
plt.savefig(PASTA / 'comparacao_por_classe.png', dpi=160)
plt.show()
''')

md('''
## 9. Curvas de treinamento e matrizes de confusão

As perdas e métricas por época ajudam a examinar o aprendizado. As matrizes
do **teste** mostram confusões entre classes, objetos perdidos e detecções extras.
Os gráficos completos também serão exportados no ZIP.
''')
codigo('''
for nome, checkpoint in CHECKPOINTS.items():
    display(Markdown(f'### {nome}'))
    arquivos = [checkpoint.parent.parent / 'results.png',
                PASTA / 'teste' / nome.lower() / 'confusion_matrix_normalized.png',
                PASTA / 'teste' / nome.lower() / 'BoxPR_curve.png']
    for arquivo in arquivos:
        if arquivo.exists():
            imagem = Image.open(arquivo).convert('RGB')
            imagem.thumbnail((1200, 900))
            display(imagem)
''')

md('''
## 10. Resultados visuais com bounding boxes

Os modelos fazem previsões em **todas** as imagens de teste, salvas com boxes,
classes e confiança. Abaixo, seis imagens escolhidas por seed antes dos treinos
mostram, lado a lado, a anotação de referência e a previsão de cada versão.
Ambos usam confiança mínima de 0,25 e o mesmo limiar de NMS (IoU 0,70).
Esse limiar de NMS é diferente dos limiares de IoU usados no cálculo do AP.

Compare caixas que faltam, caixas extras e classes confundidas. O número de
boxes desenhadas pode mudar ao alterar a confiança, sem alterar os pesos treinados.
''')
codigo('''
PASTA_VISUAL = PASTA / 'visuais'
PASTA_VISUAL.mkdir()
PREVISOES = {}
selecionadas = {str(p) for p in AMOSTRAS}
for nome, checkpoint in CHECKPOINTS.items():
    modelo = YOLO(str(checkpoint))
    PREVISOES[nome] = {}
    resultados = modelo.predict(source=str(PASTA / 'test.txt'),
                                stream=True, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                                conf=CONF_VISUAL, iou=0.7, half=False,
                                save=True, save_txt=True, save_conf=True,
                                project=str(PASTA / 'previsoes'), name=nome.lower(), verbose=False)
    for resultado in resultados:
        if str(Path(resultado.path).resolve()) in selecionadas:
            PREVISOES[nome][Path(resultado.path).name] = resultado.plot()[:, :, ::-1].copy()
    del modelo
    torch.cuda.empty_cache()

for indice, caminho in enumerate(AMOSTRAS, 1):
    fig, eixos = plt.subplots(1, 3, figsize=(18, 6))
    eixos[0].imshow(imagem_anotada(caminho))
    eixos[0].set_title('Referência anotada')
    for eixo, nome in zip(eixos[1:], MODELOS):
        eixo.imshow(PREVISOES[nome][caminho.name])
        eixo.set_title(f'{nome} | confiança >= {CONF_VISUAL}')
    for eixo in eixos:
        eixo.axis('off')
    fig.suptitle(f'Teste {indice}: {caminho.name}', fontsize=10)
    plt.tight_layout()
    plt.savefig(PASTA_VISUAL / f'comparacao_{indice:02d}.jpg', dpi=140)
    plt.show()
''')

md('''
## 11. Conclusão e limites do experimento

A conclusão abaixo usa somente os números desta execução. A diferença de mAP
é expressa em pontos percentuais. Uma execução não estabelece superioridade
geral: o resultado depende da base, da seed, do orçamento de treino e do hardware.

Uma classe ausente no teste não tem sua qualidade avaliada. Mesmo após remover
duplicatas exatas, imagens muito parecidas ou da mesma cena podem permanecer
em divisões diferentes. Não foi realizado agrupamento por obra ou por câmera.
''')
codigo('''
v8 = comparacao.set_index('modelo').loc['YOLOv8n']
v11 = comparacao.set_index('modelo').loc['YOLO11n']
diferenca = float(v11['mAP50-95'] - v8['mAP50-95'])
if abs(diferenca) < 0.00005:
    resultado_principal = 'Os modelos tiveram mAP50-95 praticamente igual na precisão apresentada.'
else:
    melhor = 'YOLO11n' if diferenca > 0 else 'YOLOv8n'
    resultado_principal = f'{melhor} obteve o maior mAP50-95, com vantagem de {abs(diferenca) * 100:.2f} pontos percentuais.'
texto = f"""# Conclusão da execução

Foram treinados YOLOv8n e YOLO11n por {EPOCHS} épocas cada, com imagens de
{IMGSZ} pixels, batch {BATCH} e seed {SEED}, a partir de pesos pré-treinados em COCO.
Após a auditoria, foram usadas {len(LISTAS['train'])} imagens de treino,
{len(LISTAS['val'])} de validação e {len(LISTAS['test'])} de teste.
Os checkpoints foram selecionados na validação, antes da avaliação no teste.

O YOLOv8n obteve mAP50 de {v8['mAP50']:.4f} e mAP50-95 de {v8['mAP50-95']:.4f}.
O YOLO11n obteve mAP50 de {v11['mAP50']:.4f} e mAP50-95 de {v11['mAP50-95']:.4f}.
{resultado_principal}

Na GPU {AMBIENTE['gpu']}, a etapa de inferência medida pelo validador levou
{v8['inferencia_ms']:.2f} ms/imagem para YOLOv8n e {v11['inferencia_ms']:.2f}
ms/imagem para YOLO11n, com batch 1 e FP32, sem pré/pós-processamento.
Esses tempos são indicativos desta execução, sem repetições de benchmark.

As imagens com boxes permitem confrontar as previsões com as anotações de referência.
O AP por classe mostra diferenças que a média geral pode esconder.
Foi feita uma execução por versão; não foram estimadas variância ou significância
estatística. O experimento se limita à base Construction-PPE e às suas anotações.
"""
ausentes = [NOMES[i] for i, n in enumerate(CONTAGENS['test']) if n == 0]
if ausentes:
    texto += '\\nClasses sem boxes no teste, sem avaliação específica: ' + ', '.join(ausentes) + '.\\n'
(PASTA / 'conclusao.md').write_text(texto, encoding='utf-8')
display(Markdown(texto))
''')

md('''
## 12. Exportação e entrega

O ZIP inclui pesos treinados, tabelas CSV, gráficos, matrizes, boxes de todas as
imagens de teste, exemplos lado a lado, auditoria da base, configuração e conclusão.
Não inclui as imagens originais completas da base.

**Entrega ao professor:** salve este notebook com todas as saídas visíveis,
compartilhe o link com acesso de leitura e confira o acesso em janela anônima.
Guarde o ZIP como cópia dos resultados. O compartilhamento do Colab é a entrega
por link; baixar o ZIP sozinho não cria esse link.
''')
codigo('''
ARQUIVO_ZIP = Path(shutil.make_archive(str(PASTA.parent / 'entrega'), 'zip', root_dir=PASTA))
print(f'Arquivo de resultados: {ARQUIVO_ZIP} ({ARQUIVO_ZIP.stat().st_size / 1e6:.1f} MB)')
try:
    from google.colab import files
except ImportError:
    display(FileLink(str(ARQUIVO_ZIP)))
else:
    files.download(str(ARQUIVO_ZIP))
''')

md('''
## Referências

1. [Guia Ultralytics indicado no enunciado](https://docs.ultralytics.com/pt/quickstart/).
2. [Construction-PPE: base, autores, classes e licença AGPL-3.0](https://docs.ultralytics.com/datasets/detect/construction-ppe/).
3. [YOLOv8](https://docs.ultralytics.com/models/yolov8/) e [YOLO11](https://docs.ultralytics.com/models/yolo11/).
4. [Treinamento](https://docs.ultralytics.com/modes/train/), [validação](https://docs.ultralytics.com/modes/val/) e [predição](https://docs.ultralytics.com/modes/predict/).
5. [Configuração da base na versão usada](https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/cfg/datasets/construction-ppe.yaml).

Atribuição da base, conforme a documentação: Mrunmayee Dalvi, Niyati Singh,
Sahil Bhingarde e Ketaki Chalke. *Construction-PPE: Personal Protective Equipment
Detection Dataset*, versão 1.0.0, 2025, Ultralytics.
''')


def resultados_salvos():
    pasta = RAIZ / 'resultados'
    if not (pasta / 'comparacao.csv').is_file():
        return []
    with (pasta / 'comparacao.csv').open(encoding='utf-8') as arquivo:
        linhas = list(csv.DictReader(arquivo))
    tabela = ['| Modelo | Precisão | Recall | mAP50 | mAP50-95 | Inferência |',
              '|---|---:|---:|---:|---:|---:|']
    for linha in linhas:
        valores = [f"{float(linha[k]) * 100:.2f}%" for k in ['precisao', 'recall', 'mAP50', 'mAP50-95']]
        tabela.append('| ' + ' | '.join([linha['modelo'], *valores, f"{float(linha['inferencia_ms']):.2f} ms"]) + ' |')
    introducao = ('## Resultados da execução salva\n\n'
                  'Execução realizada no Google Colab em 30/10/2026, com Tesla T4 e 30 épocas por modelo. '
                  'As tabelas e imagens desta seção foram importadas dos arquivos reais de `entrega.zip`, '
                  'enviado após a execução. Não é necessário treinar novamente para consultar estes resultados. '
                  'Os logs de execução das células não estão neste arquivo.\n\n' + '\n'.join(tabela) +
                  '\n\nAvaliação em 141 imagens de teste. Inferência medida pelo validador, '
                  'batch 1 e FP32, sem pré e pós-processamento.\n')
    resultado = [{'cell_type': 'markdown', 'id': 'resultado-tabela', 'metadata': {}, 'source': introducao},
                 {'cell_type': 'markdown', 'id': 'resultado-conclusao', 'metadata': {},
                  'source': (pasta / 'conclusao.md').read_text(encoding='utf-8').replace('# Conclusão da execução', '### Conclusão da execução', 1)}]
    figuras = [('comparacao.png', 'Métricas e tempo de inferência no teste'),
               ('comparacao_por_classe.png', 'AP50-95 por classe')]
    figuras += [(f'visuais/comparacao_{i:02d}.jpg', f'Boxes no teste: exemplo {i}. Referência, YOLOv8n e YOLO11n.') for i in range(1, 7)]
    for indice, (caminho, titulo) in enumerate(figuras):
        arquivo = pasta / caminho
        mime = 'image/jpeg' if arquivo.suffix == '.jpg' else 'image/png'
        resultado.append({'cell_type': 'markdown', 'id': f'resultado-figura-{indice}', 'metadata': {},
                          'source': f'### {titulo}\n\n![{titulo}](attachment:{arquivo.name})\n',
                          'attachments': {arquivo.name: {mime: base64.b64encode(arquivo.read_bytes()).decode('ascii')}}})
    return resultado


def gerar():
    notebook = {'cells': celulas[:1] + resultados_salvos() + celulas[1:],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                             'colab': {'name': 'yolo_epi.ipynb', 'provenance': []}},
                'nbformat': 4, 'nbformat_minor': 5}
    destino = RAIZ / 'yolo_epi.ipynb'
    destino.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Gerado: {destino} ({len(notebook["cells"])} células)')


if __name__ == '__main__':
    gerar()
