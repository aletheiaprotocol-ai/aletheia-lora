# arXiv v2 Patch Plan

Recommended v2 scope: metadata/link cleanup only, not a new scientific claim.

Add a short code availability paragraph near the end of the paper:

```tex
\paragraph{Code Availability.}
The official Aletheia-LoRA implementation is available at
\url{https://github.com/aletheiaprotocol-ai/aletheia-lora}.
A Hugging Face method card is available at
\url{https://huggingface.co/aletheiaprotocol/aletheia-lora}.
The released package exposes gradient probing, gradient-ranked layer selection,
and selected-layer PEFT LoRA configuration with optional asymmetric attention/MLP
rank allocation.
```

Also add the URLs to any camera-ready artifact checklist.

Do not change the scientific claim unless a new experiment is added.
