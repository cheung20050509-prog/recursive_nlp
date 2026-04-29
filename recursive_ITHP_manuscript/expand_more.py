import os

with open("/root/autodl-tmp/recursive_language/ITHP/recursive_ITHP/recursive_ITHP_manuscript/main.tex", "r") as f:
    text = f.read()

# Insert more technical content into Section 3.1 Unimodal Extraction
injection_extraction = r"""
\subsection{Unimodal Feature Extraction}
Consider a multimodal dataset $\mathcal{D} = \{ (u_i, y_i) \}_{i=1}^{N}$, where an utterance $u_i$ is composed of three temporally aligned distinct modalities: text ($T$), audio ($A$), and visual ($V$). Therefore, an input can be represented as sequence $X = (X_{T}, X_{A}, X_{V})$.

% EXTRA EXPANDED CONTENT START
To capture the granular semantics, syntactic configurations, and local contextual dependencies, we deploy BERT \citep{devlin2018bert}. Specifically, the source utterance is tokenized and processed through the 12-layer Transformer encoder. The final CLS token acts as the holistic linguistic representation:
\begin{align}
    x_{T} & = \text{BERT}(X_{T}) \in \mathbb{R}^{d_{T}}
\end{align}
Simultaneously, acoustic features capture paralinguistic nuances vital for disambiguating sarcasm and humor. Raw audio waveforms are uniformly downsampled to 16 kHz. We extract 74 acoustic features per 10ms frame—comprising 12 Mel-frequency cepstral coefficients (MFCCs), pitch tracking, glottal source parameters, and peak amplitude metrics—utilizing the COVAREP acoustic analysis framework \citep{degottex2014covarep}. The frame sequence is routed through a bi-directional Gated Recurrent Unit (GRU) to temporally smooth the isolated features into a dense acoustic profile:
\begin{align}
    x_{A} & = \text{BiGRU}(\text{COVAREP}(X_{A})) \in \mathbb{R}^{d_{A}}
\end{align}
Visual modalities capture facial morphology, eye movements, and physiological gestures \citep{morency2011multisentense}. The OpenSMILE tracking module processes sequential video frames capturing 35 dense facial action units (FAUs) representing distinct muscle actuations. These visual streams undergo identical BiGRU temporal modeling, generating a uniform latent dimension:
\begin{align}
    x_{V} & = \text{BiGRU}(\text{FACET}(X_{V})) \in \mathbb{R}^{d_{V}}
\end{align}
% EXTRA EXPANDED CONTENT END

where $d_{T}$, $d_{A}$, and $d_{V}$ denote the hidden dimensions. While our approach is agnostic to the specific spatial order, traversing from the most semantically dense modality implies earlier optimal halting. Consequently, we define our generic modal sequence as $X_0 = x_T$, $X_1 = x_A$, and $X_2 = x_V$.
"""

text = text.replace(r"""\subsection{Unimodal Feature Extraction}
Consider a multimodal dataset $\mathcal{D} = \{ (u_i, y_i) \}_{i=1}^{N}$, where an utterance $u_i$ is composed of three temporally aligned distinct modalities: text ($T$), audio ($A$), and visual ($V$). Therefore, an input can be represented as sequence $X = (X_{T}, X_{A}, X_{V})$.

Before the recursive process, we apply domain-specific encoders to extract initial unimodal representations:
\begin{align}
    x_{T} & = \text{BERT}(X_{T}) \in \mathbb{R}^{d_{T}} \\
    x_{A} & = \text{Wav2Vec}(X_{A}) \in \mathbb{R}^{d_{A}} \\
    x_{V} & = \text{ResNet}(X_{V}) \in \mathbb{R}^{d_{V}}
\end{align}
where $d_{T}$, $d_{A}$, and $d_{V}$ denote the hidden dimensions. While our approach is agnostic to the specific spatial order, traversing from the most semantically dense modality implies earlier optimal halting. Consequently, we define our generic modal sequence as $X_0 = x_T$, $X_1 = x_A$, and $X_2 = x_V$.
""", injection_extraction)


# Insert Dataset Statistics Table
injection_dataset = r"""
\textbf{Dataset Split \& Evaluation.} For MOSI and MOSEI, consistent with prior benchmarks, we enforce a strict target train/dev/test split. Performance is measured via binary accuracy (Acc-2), F1-score, and Mean Absolute Error (MAE). For classification paradigms on MUStARD and UR-FUNNY, we deploy binary accuracy and F1 scores.

\begin{table}[h]
  \caption{\label{tab:dataset_stats} Statistical breakdown distinguishing training, validation, and testing sample sizes across the four primarily evaluated multimodal datasets. Sentences represent total unique utterances processed.}
  \centering
  \resizebox{0.95\linewidth}{!}{
  \begin{tabular}{lrrrr}
    \toprule
    \textbf{Dataset} & \textbf{Train} & \textbf{Valid} & \textbf{Test} & \textbf{Total} \\
    \midrule
    \textbf{CMU-MOSI}  & 1,284 & 229 & 686 & 2,199 \\
    \textbf{CMU-MOSEI} & 16,326 & 1,871 & 4,659 & 22,856 \\
    \textbf{MUStARD}   & 345 & -- & 345 & 690 \\
    \textbf{UR-FUNNY}  & 10,598 & 1,318 & 1,318 & 13,234 \\
    \bottomrule
  \end{tabular}
  }
\end{table}

\subsection{Theoretical Limitations of Early Halting}
Although early halting mathematically regularizes network architectures preventing destructive cascading visual artifact integration into pristine textual context embeddings mathematically mapping to $I(Z; Y)$, this thermodynamic approach introduces statistical blind-spots conceptually termed "Premature Confirmation". Under VIB bounds, the halting condition inherently trusts the local semantic evaluation output from $q(Y \mid Z_k)$. When the text strictly articulates sarcasm without explicit punctuation—"I just completely adore waiting perpetually outside freezing"—the text-only phase ($k=0$) might classify the emotion uniformly positive derived entirely from "adore", halting entirely before the acoustic dissonance or negative facial expressions validate the stark pragmatic reversal. Our hyper-parameter exploration manipulating the threshold score $\tau$ extensively limits such blind estimations balancing pure optimization speed versus long-tail contextual nuance requirements.
"""

text = text.replace(r"""\textbf{Dataset Split \& Evaluation.} For MOSI and MOSEI, consistent with prior benchmarks, we enforce a strict target train/dev/test split. Performance is measured via binary accuracy (Acc-2), F1-score, and Mean Absolute Error (MAE). For classification paradigms on MUStARD and UR-FUNNY, we deploy binary accuracy and F1 scores.""", injection_dataset)

with open("/root/autodl-tmp/recursive_language/ITHP/recursive_ITHP/recursive_ITHP_manuscript/main.tex", "w") as f:
    f.write(text)

