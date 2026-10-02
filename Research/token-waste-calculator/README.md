# Token Waste Calculator (side artifact)

Runs on the Mac. No GPU. Can ship during paper 1.

Plug into LangChain / LlamaIndex / custom traces and report:

1. Share of prompts that are deterministic extraction / schema fill (should not hit a frontier API).
2. Estimated annual $ if those calls moved to a 1B–8B distilled model.

CLI lives here later (`astrolab-audit`). Do not block CUAD QLoRA on this.
