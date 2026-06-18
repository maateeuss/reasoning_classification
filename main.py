import os
import re
import subprocess
import threading
import time
import glob
import html
from datetime import datetime
from collections import Counter

import requests
import pdfplumber
from dateparser.search import search_dates
from colorama import init, Fore, Style
import csv
from tqdm import tqdm

init(autoreset=True)

# ==================== CONFIGURAÇÕES ====================
BASE_DIR = r"E:\reasoning_classification"
AMOSTRA_DIR = os.path.join(BASE_DIR, "Amostra")
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-IQ3_XS.gguf")
LLAMA_VULKAN = os.path.join(BASE_DIR, "llama-server", "llama-b9494-bin-win-vulkan-x64", "llama-server.exe")
LLAMA_CPU = os.path.join(BASE_DIR, "llama-server", "llama-b9484-bin-win-cpu-x64", "llama-server.exe")
PORT = 8080
API_URL = f"http://localhost:{PORT}/v1/chat/completions"
CHUNK_SIZE = 4096          # caracteres
STEP = 500                 # deslocamento para trás
MAX_CHUNKS = 5             # número máximo de chunks a processar

MAX_TOKENS = 3072
CTX_SIZE = 4096

DATE_CONFIG = {
    'DATE_ORDER': 'DMY',
    'SKIP_TOKENS': ['de', 'do', 'da', 'a', 'as']
}

# ==================== FUNÇÕES DE LOG ====================
def log_info(msg):
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {msg}")

def log_success(msg):
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {msg}")

def log_warning(msg):
    print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {msg}")

def log_error(msg):
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")

def log_prompt(msg):
    print(f"{Fore.BLUE}[PROMPT]{Style.RESET_ALL} {msg}")

def log_resposta(msg):
    print(f"{Fore.YELLOW}[RESPOSTA]{Style.RESET_ALL} {msg}")

# ==================== LIMPEZA DE PDF ====================
def limpar_texto_pdf(texto: str) -> str:
    header_pattern = re.compile(
        r"^.*?TRIBUNAL DE JUSTIÇA DO ESTADO DE SÃO PAULO.*?"
        r"(?=\b(?:DECISÃO|SENTENÇA|DESPACHO|CERTIDÃO|ATA|PROCLAMA|MANDADO|OFÍCIO|CITAÇÃO|INTIMAÇÃO)\b|\bProcesso\s+(?:Digital\s+)?n\.?º?:)",
        re.IGNORECASE | re.DOTALL
    )
    texto = header_pattern.sub("", texto)
    texto = re.sub(r"^\s*\d{7}-\d{2}\.\d{4}\.\d{2}\.\d{4}\s*[-–]?\s*(?:lauda\s*\d+)?\s*", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"P\s*a\s*r\s*a\s*c\s*o\s*n\s*f\s*e\s*r\s*i\s*r.*?fls\.\s*\d+\s*", "", texto, flags=re.DOTALL | re.IGNORECASE)
    texto = re.sub(r"E\s*s\s*t\s*e\s*d\s*o\s*c\s*u\s*m\s*e\s*n\s*t\s*o\s*é\s*c.*?fls\.\s*\d+\s*", "", texto, flags=re.DOTALL | re.IGNORECASE)
    texto = re.sub(r"DOCUMENTO ASSINADO DIGITALMENTE NOS TERMOS DA LEI 11\.419/2006.*?MARGEM DIREITA\s*", "", texto, flags=re.DOTALL | re.IGNORECASE)
    texto = re.sub(r"\bfls\.\s*\d+\s*", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    paragrafos = texto.split('\n\n')
    paragrafos_limpos = []
    for p in paragrafos:
        p_limpo = re.sub(r'\s*\n\s*', ' ', p).strip()
        p_limpo = re.sub(r' +', ' ', p_limpo)
        if p_limpo:
            paragrafos_limpos.append(p_limpo)
    return '\n\n'.join(paragrafos_limpos)

def extrair_numero_arquivo(caminho: str) -> int:
    nome = os.path.basename(caminho)
    match = re.search(r'sub_(\d+)', nome)
    return int(match.group(1)) if match else 0

def extrair_texto_pdf(caminho: str) -> str:
    """Extrai e limpa o texto completo do PDF (sem truncamento)"""
    try:
        with pdfplumber.open(caminho) as pdf:
            texto = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return limpar_texto_pdf(texto)
    except Exception as e:
        log_error(f"Erro ao extrair texto de {caminho}: {e}")
        return ""

def extrair_datas_do_texto(texto: str):
    try:
        resultados = search_dates(texto, languages=['pt'], settings=DATE_CONFIG)
        if resultados:
            return [dt for _, dt in resultados]
    except Exception as e:
        log_error(f"Erro no dateparser: {e}")
    return []

def obter_data_mais_recente_pdf(caminho: str):
    """Retorna (data, metodo, chave) para ordenação"""
    texto = extrair_texto_pdf(caminho)
    datas = extrair_datas_do_texto(texto)
    if datas:
        current_year = datetime.now().year
        datas_naive = []
        for dt in datas:
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            if 1900 <= dt.year <= current_year:
                datas_naive.append(dt)
        if datas_naive:
            data_max = max(datas_naive)
            return data_max, 'texto', None
        else:
            log_warning(f"  Nenhuma data com ano entre 1900 e {current_year} após filtro.")
    numero = extrair_numero_arquivo(caminho)
    if numero > 0:
        return None, 'numero', numero
    mtime = os.path.getmtime(caminho)
    return datetime.fromtimestamp(mtime), 'timestamp', mtime

def determinar_tipo_documento(caminho: str) -> int:
    """
    Retorna peso do documento:
    4 = sentença/decisão/acórdão
    2 = despacho
    1 = certidão
    0 = outros
    """
    nome = os.path.basename(caminho).lower()
    # Primeiro tenta pelo nome do arquivo
    if any(palavra in nome for palavra in ['sentença', 'decisão', 'decisao', 'acórdão', 'acordao']):
        return 4
    if 'despacho' in nome:
        return 2
    if 'certidão' in nome or 'certidao' in nome:
        return 1
    # Se não encontrou pelo nome, lê as primeiras 1000 letras do texto
    texto = extrair_texto_pdf(caminho)[:1000].lower()
    if any(palavra in texto for palavra in ['sentença', 'decisão', 'decisao', 'acórdão', 'acordao']):
        return 4
    if 'despacho' in texto:
        return 2
    if 'certidão' in texto or 'certidao' in texto:
        return 1
    return 0

# ==================== EXTRAÇÃO DAS PARTES DO HTML ====================
def extrair_partes_html(caminho_html: str) -> str:
    try:
        with open(caminho_html, 'r', encoding='utf-8') as f:
            conteudo = f.read()
    except Exception as e:
        log_warning(f"Não foi possível ler o HTML {caminho_html}: {e}")
        return ""

    match = re.search(r'<table[^>]*id="tablePartesPrincipais"[^>]*>(.*?)<tr>', conteudo, re.DOTALL | re.IGNORECASE)
    if not match:
        log_warning(f"Tabela de partes não encontrada no HTML {caminho_html}")
        return ""

    table_html = match.group(1)
    rows = re.findall(r'<tr\b[^>]*>(.*?)</td>', table_html, re.DOTALL | re.IGNORECASE)
    linhas = []

    for row_html in rows:
        cells = re.findall(r'<td\b[^>]*>(.*?)</td>', row_html, re.DOTALL | re.IGNORECASE)
        if not cells:
            continue
        cell_texts = []
        for cell in cells:
            cell_clean = re.sub(r'<[^>]+>', ' ', cell)
            cell_clean = html.unescape(cell_clean)
            cell_clean = re.sub(r'\s+', ' ', cell_clean).strip()
            if cell_clean:
                cell_texts.append(cell_clean)
        if cell_texts:
            linhas.append(' '.join(cell_texts))

    return '\n'.join(linhas)

# ==================== SERVIDOR LLAMA ====================
def log_output(proc, name):
    for line in iter(proc.stdout.readline, b''):
        if line:
            print(f"[{name}] {line.decode('utf-8', errors='ignore').strip()}")

def iniciar_servidor():
    try:
        r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
        if r.status_code == 200:
            log_info(f"Servidor já está rodando na porta {PORT}. Reutilizando.")
            return None
    except:
        pass

    if os.path.exists(LLAMA_VULKAN):
        exe = LLAMA_VULKAN
        gpu_layers = 99
        log_info("Usando Vulkan (iGPU)")
    else:
        exe = LLAMA_CPU
        gpu_layers = 0
        log_info("Usando CPU (Vulkan não encontrado)")

    cmd = f'"{exe}" --model "{MODEL_PATH}" --ctx-size {CTX_SIZE} --n-gpu-layers {gpu_layers} --port {PORT} --temp 0.0 --top-k 1 --cache-ram 0 --parallel 1 --cont-batching'
    log_info(f"Iniciando servidor: {cmd}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    threading.Thread(target=log_output, args=(proc, "LLAMA"), daemon=True).start()

    for _ in range(300):
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                log_success("Servidor pronto!")
                return proc
        except:
            pass
        time.sleep(1)
    log_error("Servidor não respondeu a tempo.")
    return None

def encerrar_servidor(proc):
    if proc:
        proc.terminate()
        proc.wait(timeout=10)
        log_info("Servidor encerrado.")

# ==================== CLASSIFICAÇÃO DE UM TEXTO (API) ====================
def classificar_texto(texto: str, partes_texto: str = "") -> tuple[int | None, str]:
    """Retorna (label, resposta_bruta) para um único chunk. label pode ser None se falhar."""
    system_msg = (
        "Você é um assistente jurídico. Classifique a decisão judicial em relação ao réu conforme as regras abaixo.\n"
        "Responda EXCLUSIVAMENTE com uma linha no formato: CLASSIFICACAO: X\n"
        "onde X pode ser -1, 0 ou 1.\n"
        "Não inclua nenhum outro texto, explicação ou raciocínio.\n\n"
        "Definições:\n"
        "-1: O juiz decide contra o réu (condena, nega recurso, mantém pena, impõe multa, etc.)\n"
        "0: O juiz não decide sobre o mérito da acusação. Exemplos: despachos de andamento, redistribuições, prazos, suspensões, extinção do processo por pagamento da dívida ou cumprimento da obrigação antes da sentença.\n"
        "1: O juiz decide a favor do réu (absolve, extingue a pena por indulto/amnistia, arquiva o inquérito sem possibilidade de reabertura, etc.)"
    )

    partes_bloco = f"Partes do processo:\n{partes_texto}\n\n" if partes_texto else ""

    user_msg = f"""{partes_bloco}Texto da decisão:
{texto}"""

    payload = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "stop": [],
        "stream": False,
    }

    log_prompt(f"Prompt (primeiros 2000 chars):\n{user_msg[:2000]}")
    if len(user_msg) > 2000:
        log_prompt(f"... total {len(user_msg)} caracteres")

    try:
        resp = requests.post(API_URL, json=payload, timeout=900)
        resp.raise_for_status()
        resultado = resp.json()
        choices = resultado.get("choices", [])
        if not choices:
            log_error("Resposta sem 'choices'.")
            return None, ""

        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        reasoning = message.get("reasoning_content", "").strip()
        resposta_bruta = (reasoning + "\n" + content) if reasoning else content

        log_resposta(f"Pensamento (reasoning):\n{reasoning}")
        log_resposta(f"Resposta final (content):\n{content}")
        log_resposta(f"Texto combinado (CSV):\n{resposta_bruta}")

        # Se a resposta final estiver vazia, é falha
        if not content:
            log_error("Resposta final vazia.")
            return None, resposta_bruta

        # Extrai o label apenas do content (resposta final)
        texto_resposta = content
        # Remove possíveis tags de pensamento
        if '</think>' in texto_resposta:
            texto_resposta = texto_resposta.split('</think>', 1)[1].strip()
        if '<think>' in texto_resposta:
            texto_resposta = texto_resposta.split('<think>', 1)[0].strip()

        # Padrões flexíveis
        patterns = [
            r'CLASSIFICACAO\s*:\s*(-1|0|1)',
            r'classificacao\s*:\s*(-1|0|1)',
            r'"classificacao"\s*:\s*(-1|0|1)',
            r'(-1|0|1)'   # fallback
        ]
        for pattern in patterns:
            m = re.search(pattern, texto_resposta)
            if m:
                label = int(m.group(1))
                log_success(f"Classificação extraída: {label}")
                return label, resposta_bruta

        log_error(f"Não foi possível extrair label. Resposta: {texto_resposta[:200]}")
        return None, resposta_bruta

    except Exception as e:
        log_error(f"Erro na chamada da API: {e}")
        return None, ""

# ==================== CLASSIFICAÇÃO COM JANELA DESLIZANTE E CONSISTÊNCIA ====================
def classificar_com_janela(texto_completo: str, partes_texto: str = "") -> tuple[int | None, str, str]:
    """
    Retorna (label, resposta_bruta, erro_descricao)
    erro_descricao pode ser: "", "Falha na classificação", "Indeterminado (divergência entre chunks)", "Inconsistência entre chunks"
    """
    if not texto_completo.strip():
        return None, "", "Falha na classificação (texto vazio)"

    # Se o texto é curto, classifica diretamente
    if len(texto_completo) <= CHUNK_SIZE:
        label, resposta = classificar_texto(texto_completo, partes_texto)
        if label is not None:
            return label, resposta, ""
        else:
            return None, resposta, "Falha na classificação"

    # Texto longo: processa chunks a partir do final
    inicio = max(0, len(texto_completo) - CHUNK_SIZE)
    labels = []
    respostas = []
    chunks_processados = 0

    while inicio >= 0 and chunks_processados < MAX_CHUNKS:
        chunk = texto_completo[inicio:inicio + CHUNK_SIZE]
        log_info(f"Processando chunk de tamanho {len(chunk)} (início={inicio})")
        label, resp = classificar_texto(chunk, partes_texto)
        if label is not None:
            labels.append(label)
            respostas.append(resp)
        chunks_processados += 1
        if inicio == 0:
            break
        inicio = max(0, inicio - STEP)

    if not labels:
        return None, respostas[0] if respostas else "", "Falha na classificação"

    # Análise de consistência com regras prioritárias
    if len(labels) == 1:
        return labels[0], respostas[0], ""

    # Verifica se há conflito entre 0 e não-zero
    set_labels = set(labels)

    # Regra: 0 vs 1 -> 1
    if 0 in set_labels and 1 in set_labels and -1 not in set_labels:
        # Apenas 0 e 1 presentes
        return 1, respostas[0], "Inconsistência entre chunks (0 vs 1 -> 1)"

    # Regra: 0 vs -1 -> -1
    if 0 in set_labels and -1 in set_labels and 1 not in set_labels:
        return -1, respostas[0], "Inconsistência entre chunks (0 vs -1 -> -1)"

    # Caso haja 1 e -1 juntos (com ou sem 0)
    if 1 in set_labels and -1 in set_labels:
        # Opção 1: considerar indeterminado
        return None, respostas[0], "Indeterminado (1 vs -1)"
        # Opção 2 (se preferir -1): descomente a linha abaixo e comente a de cima
        # return -1, respostas[0], "Inconsistência entre chunks (1 vs -1 -> -1)"

    # Se todos iguais (incluindo só 0, só 1 ou só -1)
    if len(set_labels) == 1:
        return labels[0], respostas[0], ""

    # Fallback (não deveria acontecer)
    return None, respostas[0], "Divergência não classificável"

# ==================== PROCESSAMENTO PRINCIPAL ====================
def main():
    log_info("=== Classificador de Decisões Judiciais ===")

    if not os.path.exists(MODEL_PATH):
        log_error(f"Modelo não encontrado: {MODEL_PATH}")
        return

    server_proc = iniciar_servidor()
    if server_proc is None:
        try:
            requests.get(f"http://localhost:{PORT}/health", timeout=5)
        except:
            log_error("Não foi possível conectar ao servidor existente.")
            return

    csv_path = os.path.join(BASE_DIR, "resultados_classificacao.csv")
    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["subpasta", "pdf_escolhido", "data_usada", "metodo_data",
                         "partes", "label", "erro", "resposta_completa"])

    subpastas = [d for d in glob.glob(os.path.join(AMOSTRA_DIR, "*")) if os.path.isdir(d)]
    log_info(f"Encontradas {len(subpastas)} subpastas.")

    for idx, pasta in enumerate(tqdm(subpastas, desc="Processando"), 1):
        nome_pasta = os.path.basename(pasta)
        log_info(f"\n[{idx}/{len(subpastas)}] Processando: {nome_pasta}")

        # 1. Extrair partes do HTML
        html_file = os.path.join(pasta, "principal.html")
        partes_texto = ""
        if os.path.exists(html_file):
            partes_texto = extrair_partes_html(html_file)
            if partes_texto:
                log_info(f"  Partes extraídas: {partes_texto[:150]}{'...' if len(partes_texto)>150 else ''}")
            else:
                log_warning("  Partes extraídas estão vazias.")
        else:
            log_warning(f"  HTML principal.html não encontrado em {pasta}")

        # 2. Listar todos os PDFs e enriquecer com data, tipo, texto
        pdfs = glob.glob(os.path.join(pasta, "*.pdf"))
        if not pdfs:
            log_warning(f"  Nenhum PDF encontrado. Pulando.")
            with open(csv_path, "a", encoding="utf-8", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([nome_pasta, "", "", "", partes_texto, "", "Nenhum PDF", ""])
            continue

        # Para cada PDF, obter data, tipo e texto completo (para possível concatenação)
        info_pdfs = []
        for pdf in pdfs:
            data_obj, metodo, chave = obter_data_mais_recente_pdf(pdf)
            peso = determinar_tipo_documento(pdf)
            # Converte data_obj para string de ordenação (datetime ou None)
            data_ord = data_obj if data_obj else datetime.min
            texto_completo = extrair_texto_pdf(pdf)  # sem truncamento
            info_pdfs.append({
                'caminho': pdf,
                'data_obj': data_obj,
                'metodo': metodo,
                'chave': chave,
                'peso': peso,
                'data_ord': data_ord,
                'texto': texto_completo
            })

        # Ordenar por peso desc, data desc
        info_pdfs.sort(key=lambda x: (-x['peso'], -x['data_ord'].timestamp() if x['data_ord'] else 0))

        # Selecionar o(s) primeiro(s) com mesmo peso e mesma data (tolerância 1 segundo)
        primeiro = info_pdfs[0]
        selecionados = [primeiro]
        for outro in info_pdfs[1:]:
            if outro['peso'] == primeiro['peso']:
                # Compara datas: se ambos têm data_obj e diferem menos de 1 segundo
                if primeiro['data_obj'] and outro['data_obj']:
                    diff = abs((primeiro['data_obj'] - outro['data_obj']).total_seconds())
                    if diff < 1:
                        selecionados.append(outro)
                    else:
                        break
                else:
                    # Se um não tem data, não consideramos igual
                    break
            else:
                break

        # Concatenar textos dos selecionados (ordem da mais recente para mais antiga)
        selecionados.sort(key=lambda x: x['data_ord'] if x['data_ord'] else datetime.min, reverse=True)
        texto_concatenado = "\n\n---\n\n".join([s['texto'] for s in selecionados if s['texto']])

        if not texto_concatenado:
            log_error("  Texto extraído vazio para o(s) PDF(s) selecionado(s).")
            with open(csv_path, "a", encoding="utf-8", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([nome_pasta, " | ".join([s['caminho'] for s in selecionados]), "", "", partes_texto, "", "Texto vazio", ""])
            continue

        # Data usada: a maior data entre os selecionados
        datas_validas = [s['data_obj'] for s in selecionados if s['data_obj']]
        if datas_validas:
            data_usada = max(datas_validas).strftime("%Y-%m-%d")
            metodo_data = selecionados[datas_validas.index(max(datas_validas))]['metodo']
        else:
            # Fallback para o primeiro método (numero ou timestamp)
            data_usada = str(selecionados[0]['chave']) if selecionados[0]['chave'] is not None else "desconhecida"
            metodo_data = selecionados[0]['metodo']

        log_info(f"  PDF(s) selecionado(s): {[s['caminho'] for s in selecionados]}")
        log_info(f"  Data/Método: {data_usada} ({metodo_data})")
        log_info(f"  Texto concatenado tem {len(texto_concatenado)} caracteres")

        # 3. Classificar com janela deslizante e consistência
        label, resposta_bruta, erro = classificar_com_janela(texto_concatenado, partes_texto)

        # 4. Escrever no CSV (uma linha por processo)
        pdfs_str = " | ".join([s['caminho'] for s in selecionados])
        label_str = str(label) if label is not None else ""
        resposta_csv = resposta_bruta.replace('\n', ' ').replace('\r', ' ') if resposta_bruta else ""

        with open(csv_path, "a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([nome_pasta, pdfs_str, data_usada, metodo_data, partes_texto, label_str, erro, resposta_csv])

        if erro:
            log_warning(f"  Classificação com erro: {erro}")
        else:
            log_success(f"  Classificação: {label}")

    log_info("\nProcessamento concluído!")
    log_info(f"Resultados salvos em: {csv_path}")

    if server_proc is not None:
        encerrar_servidor(server_proc)

if __name__ == "__main__":
    main()