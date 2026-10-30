"""Auditoria das anotações e remoção de duplicatas exatas entre divisões."""

import hashlib
import math
from pathlib import Path


def ler_boxes(arquivo, numero_classes):
    """Lê classe, centro x/y e largura/altura normalizados no formato YOLO."""
    boxes = []
    for numero, linha in enumerate(Path(arquivo).read_text(encoding='utf-8').splitlines(), 1):
        if not linha.strip():
            continue
        try:
            valores = [float(item) for item in linha.split()]
            if len(valores) != 5 or not all(math.isfinite(v) for v in valores):
                raise ValueError('esperados cinco números finitos')
            classe, x, y, largura, altura = valores
            if not classe.is_integer() or not 0 <= classe < numero_classes:
                raise ValueError('classe inválida')
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < largura <= 1 and 0 < altura <= 1):
                raise ValueError('coordenadas inválidas')
            boxes.append((int(classe), x, y, largura, altura))
        except ValueError as erro:
            raise ValueError(f'{arquivo}, linha {numero}: {erro}') from erro
    return boxes


def arquivo_label(imagem):
    partes = list(Path(imagem).parts)
    indice = len(partes) - 1 - partes[::-1].index('images')
    partes[indice] = 'labels'
    return Path(*partes).with_suffix('.txt')


def preparar_divisoes(divisoes, numero_classes):
    """Mantém a primeira ocorrência de cada arquivo na ordem train, val, test.

    Não altera os originais. SHA-256 compara bytes, não imagens visualmente parecidas.
    As listas retornadas são usadas por ambos os modelos.
    """
    vistos, listas, resumo, duplicatas, contagens = {}, {}, {}, [], {}
    extensoes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for split in ['train', 'val', 'test']:
        imagens = sorted(p for p in Path(divisoes[split]).rglob('*') if p.suffix.lower() in extensoes)
        listas[split], contagens[split] = [], [0] * numero_classes
        for imagem in imagens:
            digest = hashlib.sha256(imagem.read_bytes()).hexdigest()
            if digest in vistos:
                duplicatas.append({'excluida': str(imagem), 'mantida': vistos[digest], 'sha256': digest})
                continue
            boxes = ler_boxes(arquivo_label(imagem), numero_classes)
            vistos[digest] = str(imagem)
            listas[split].append(imagem.resolve())
            for classe, *_ in boxes:
                contagens[split][classe] += 1
        usados = len(listas[split])
        if not usados:
            raise ValueError(f'O conjunto {split} ficou vazio após a auditoria.')
        resumo[split] = {'imagens_originais': len(imagens), 'imagens_usadas': usados,
                        'duplicatas_excluidas': len(imagens) - usados,
                        'boxes': sum(contagens[split])}
    return listas, resumo, duplicatas, contagens


def resumir_classes(modelo, nomes, contagens, indices, precisao, recall, ap50, ap):
    """Zero para classe anotada sem detecções; omite classe ausente no teste."""
    posicoes = {int(classe): i for i, classe in enumerate(indices)}
    linhas = []
    for classe in sorted(nomes):
        if not contagens[classe]:
            continue
        i = posicoes.get(classe)
        linhas.append({'modelo': modelo, 'classe': nomes[classe],
                       'boxes_teste': contagens[classe],
                       'precisao': float(precisao[i]) if i is not None else 0.0,
                       'recall': float(recall[i]) if i is not None else 0.0,
                       'AP50': float(ap50[i]) if i is not None else 0.0,
                       'AP50-95': float(ap[i]) if i is not None else 0.0})
    return linhas
