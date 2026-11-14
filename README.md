Inference serving for neural network models: one FastAPI service that exposes
OCR, speech synthesis, transcription and an OpenAI-compatible Responses path,
each backed by a registered model.

## Install

```shell
pip install nnserve
```

The `pdf` extra adds page rendering for PDF inputs:

```shell
pip install "nnserve[pdf]"
```

Kitten TTS needs its text-frontend dependencies:

```shell
pip install "nnserve[kitten]"
```

Qwen3-ASR needs the Whisper feature extractor:

```shell
pip install "nnserve[asr]"
```

## Sponsor

<table align="center">
    <thead>
        <tr>
            <th colspan="2">公众号</th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td colspan="2"><img src="https://jiauzhang.github.io/ghstatic/images/ofa_m.png" style="height: 196px" alt="AliPay.png"></td>
        </tr>
    </tbody>
    <thead>
        <tr>
            <th>AliPay</th>
            <th>WeChatPay</th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td><img src="https://jiauzhang.github.io/AliPay.png" style="width: 196px; height: 196px" alt="AliPay.png"></td>
            <td><img src="https://jiauzhang.github.io/WeChatPay.png" style="width: 196px; height: 196px" alt="WeChatPay.png"></td>
        </tr>
    </tbody>
</table>
