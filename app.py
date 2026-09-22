import re
import streamlit as st
from pypdf import PdfReader
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Configuração da página
st.set_page_config(
    page_title="Assistente Pedagógico",
    page_icon="🎓",
    layout="wide"
)


# Estilo da página
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f7fb;
        }

        .titulo-principal {
            color: #123b6d;
            font-size: 38px;
            font-weight: 800;
            margin-bottom: 0;
        }

        .subtitulo {
            color: #4b5563;
            font-size: 18px;
            margin-top: 4px;
            margin-bottom: 24px;
        }

        .aviso {
            padding: 14px;
            border-radius: 10px;
            background-color: #fff4e5;
            border-left: 5px solid #f59e0b;
            color: #5f4300;
            margin-bottom: 18px;
        }

        .rodape {
            text-align: center;
            color: #6b7280;
            font-size: 13px;
            margin-top: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def extrair_paginas(pdf_enviado):
    """Extrai o texto do PDF e preserva o número das páginas."""

    leitor = PdfReader(pdf_enviado)
    paginas = []

    for numero, pagina in enumerate(leitor.pages, start=1):
        texto = pagina.extract_text() or ""
        texto = re.sub(r"\s+", " ", texto).strip()

        if texto:
            paginas.append(
                {
                    "pagina": numero,
                    "texto": texto
                }
            )

    return paginas


def dividir_em_trechos(
    paginas,
    tamanho=1800,
    sobreposicao=250
):
    """Divide o texto do documento em partes menores."""

    trechos = []

    for item in paginas:
        texto = item["texto"]
        pagina = item["pagina"]
        inicio = 0

        while inicio < len(texto):
            fim = inicio + tamanho
            parte = texto[inicio:fim].strip()

            if parte:
                trechos.append(
                    {
                        "pagina": pagina,
                        "texto": parte
                    }
                )

            if fim >= len(texto):
                break

            inicio = fim - sobreposicao

    return trechos


def localizar_trechos(pergunta, trechos, quantidade=5):
    """Localiza os trechos mais relacionados à pergunta."""

    textos = [item["texto"] for item in trechos]

    vectorizador = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2)
    )

    matriz = vectorizador.fit_transform(
        textos + [pergunta]
    )

    vetor_pergunta = matriz[-1]
    vetores_documento = matriz[:-1]

    similaridades = cosine_similarity(
        vetor_pergunta,
        vetores_documento
    ).flatten()

    melhores_indices = similaridades.argsort()[::-1][
        :quantidade
    ]

    resultados = []

    for indice in melhores_indices:
        if similaridades[indice] > 0:
            resultados.append(
                {
                    "pagina": trechos[indice]["pagina"],
                    "texto": trechos[indice]["texto"]
                }
            )

    return resultados


def montar_contexto(trechos_encontrados):
    """Organiza os trechos que serão enviados ao Gemini."""

    partes = []

    for item in trechos_encontrados:
        partes.append(
            f"[Página {item['pagina']}]\n"
            f"{item['texto']}"
        )

    return "\n\n".join(partes)


def responder_com_gemini(pergunta, contexto):
    """Envia a pergunta e os trechos do PDF ao Gemini."""

    chave = st.secrets["GEMINI_API_KEY"]

    modelo = st.secrets.get(
        "GEMINI_MODEL",
        "gemini-2.5-flash"
    )

    cliente = genai.Client(api_key=chave)

    instrucao = f"""
Você é o Assistente Pedagógico da
EE Benedito Aparecido Tavares Prof.

Responda exclusivamente com base nos trechos do documento
apresentados abaixo.

REGRAS:

1. Não invente informações.
2. Não use informações externas ao documento.
3. Responda em português do Brasil.
4. Use linguagem clara, profissional e pedagógica.
5. Indique as páginas utilizadas.
6. Se a resposta não estiver no documento, diga:
   "Não localizei essa informação no documento enviado."
7. Preserve o sentido original do documento.
8. Não apresente dados que não estejam explicitamente registrados.

TRECHOS DO DOCUMENTO:

{contexto}

PERGUNTA:

{pergunta}
"""

    resposta = cliente.models.generate_content(
        model=modelo,
        contents=instrucao
    )

    return resposta.text


# Cabeçalho
st.markdown(
    '<p class="titulo-principal">'
    '🎓 Assistente Pedagógico'
    '</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="subtitulo">
        EE Benedito Aparecido Tavares Prof.<br>
        Consulte documentos escolares com apoio da
        Inteligência Artificial.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="aviso">
        <strong>Proteção de dados:</strong>
        não envie documentos com nomes de estudantes,
        notas, laudos, frequência, CPF, RG, endereço,
        telefone ou outras informações pessoais.
    </div>
    """,
    unsafe_allow_html=True
)


# Barra lateral
with st.sidebar:
    st.header("📄 Documento")

    arquivo_pdf = st.file_uploader(
        "Selecione um arquivo PDF",
        type=["pdf"],
        help="Utilize somente documentos sem dados pessoais."
    )

    st.divider()

    st.markdown("### Exemplos de perguntas")

    st.markdown(
        """
        - Resuma o documento.
        - Qual é o objetivo principal?
        - Quais ações são recomendadas?
        - Quais são as datas do cronograma?
        - Quais responsabilidades são apresentadas?
        - Quais habilidades curriculares aparecem?
        """
    )

    st.divider()

    if st.button(
        "🗑️ Limpar conversa",
        use_container_width=True
    ):
        st.session_state.mensagens = []
        st.rerun()


# Memória da conversa
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []


# Página inicial
if arquivo_pdf is None:
    st.info(
        "Envie um arquivo PDF na barra lateral "
        "para iniciar a consulta."
    )

    st.markdown(
        "### O que este assistente pode fazer?"
    )

    coluna1, coluna2, coluna3 = st.columns(3)

    with coluna1:
        st.markdown("#### 📑 Resumir")
        st.write(
            "Produzir resumos objetivos dos documentos."
        )

    with coluna2:
        st.markdown("#### 🔎 Localizar")
        st.write(
            "Encontrar orientações, datas, ações e "
            "responsabilidades."
        )

    with coluna3:
        st.markdown("#### 🎯 Organizar")
        st.write(
            "Transformar informações em listas e "
            "sínteses pedagógicas."
        )


# Processamento do PDF
else:
    identificador = (
        arquivo_pdf.name,
        arquivo_pdf.size
    )

    if (
        st.session_state.get("pdf_identificador")
        != identificador
    ):
        try:
            with st.spinner(
                "Processando o documento..."
            ):
                paginas = extrair_paginas(arquivo_pdf)

                if not paginas:
                    st.error(
                        "Não foi possível extrair texto deste "
                        "PDF. Ele pode ser uma digitalização "
                        "composta apenas por imagens."
                    )
                    st.stop()

                trechos = dividir_em_trechos(paginas)

                st.session_state.paginas_pdf = paginas
                st.session_state.trechos_pdf = trechos
                st.session_state.pdf_identificador = (
                    identificador
                )
                st.session_state.mensagens = []

            st.success(
                f"Documento processado: {arquivo_pdf.name}"
            )

        except Exception as erro:
            st.error(
                f"Erro ao processar o PDF: {erro}"
            )
            st.stop()

    st.caption(
        f"Documento ativo: {arquivo_pdf.name} | "
        f"Páginas com texto: "
        f"{len(st.session_state.paginas_pdf)}"
    )

    # Histórico
    for mensagem in st.session_state.mensagens:
        with st.chat_message(mensagem["papel"]):
            st.markdown(mensagem["conteudo"])

    pergunta = st.chat_input(
        "Digite uma pergunta sobre o documento..."
    )

    if pergunta:
        st.session_state.mensagens.append(
            {
                "papel": "user",
                "conteudo": pergunta
            }
        )

        with st.chat_message("user"):
            st.markdown(pergunta)

        with st.chat_message("assistant"):
            with st.spinner(
                "Consultando o documento..."
            ):
                trechos_encontrados = []

                try:
                    trechos_encontrados = localizar_trechos(
                        pergunta,
                        st.session_state.trechos_pdf
                    )

                    if not trechos_encontrados:
                        resposta = (
                            "Não localizei essa informação "
                            "no documento enviado."
                        )

                    else:
                        contexto = montar_contexto(
                            trechos_encontrados
                        )

                        resposta = responder_com_gemini(
                            pergunta,
                            contexto
                        )

                    st.markdown(resposta)

                    if trechos_encontrados:
                        paginas_usadas = sorted(
                            {
                                item["pagina"]
                                for item
                                in trechos_encontrados
                            }
                        )

                        paginas_formatadas = ", ".join(
                            str(numero)
                            for numero in paginas_usadas
                        )

                        st.caption(
                            "Trechos consultados nas páginas: "
                            f"{paginas_formatadas}"
                        )

                        with st.expander(
                            "Ver os trechos utilizados"
                        ):
                            for item in trechos_encontrados:
                                st.markdown(
                                    f"**Página "
                                    f"{item['pagina']}**"
                                )
                                st.write(item["texto"])
                                st.divider()

                except KeyError:
                    resposta = (
                        "A chave GEMINI_API_KEY ainda não foi "
                        "configurada nos segredos do aplicativo."
                    )
                    st.error(resposta)

                except Exception as erro:
                    resposta = (
                        "Não foi possível gerar a resposta. "
                        f"Detalhes técnicos: {erro}"
                    )
                    st.error(resposta)

        st.session_state.mensagens.append(
            {
                "papel": "assistant",
                "conteudo": resposta
            }
        )


# Rodapé
st.markdown(
    """
    <div class="rodape">
        Protótipo educacional •
        EE Benedito Aparecido Tavares Prof.<br>
        As respostas devem ser conferidas no documento original.
    </div>
    """,
    unsafe_allow_html=True
)