# In-Depth Summary: *Amodal Completion via Progressive Mixed Context Diffusion*

For the implementation trace, function locations, and debugging workflow, see [Paper → Python file → function](PAPER_CODE_MAP.md).

**Authors:** Katherine Xu, Lingzhi Zhang, and Jianbo Shi  
**Affiliations:** University of Pennsylvania and Adobe  
**Paper version:** arXiv v1, 24 December 2023  

## 1. Executive summary

The paper introduces a method for **amodal completion**: reconstructing the parts of an object that are hidden by other objects or cut off by the image boundary. For example, given a partly hidden bicycle, the method tries to infer and generate the bicycle’s unseen frame, wheels, or handlebars while preserving the visible bicycle and the surrounding scene.

The central idea is to use a pre-trained image-diffusion model as a powerful visual prior, but to control it carefully. Rather than asking the model to outpaint the entire area around an object—which often changes the object’s identity or creates plausible but unwanted neighboring objects—the method:

1. Detects which nearby objects actually occlude the object of interest.
2. Inpaints only those occluder regions.
3. Repeats the process until no important occlusion remains.
4. Temporarily replaces the original context with a clean background during diffusion to reduce the tendency to regenerate co-occurring objects (for example, a hand near a cup).
5. Uses a counterfactual outpainting test to estimate whether the generated object is complete.

The approach is **training-free**: it uses publicly available, pre-trained models and does not fine-tune them or require a dedicated amodal-completion training set. On a pseudo-occlusion benchmark built from natural images, it generally outperforms earlier GAN-based, inpainting, and object-removal methods. A user study strongly favors the proposed results, although the automatic completeness checker is only moderately accurate and several difficult occlusion cases remain unsolved.

## 2. What problem is being solved?

### 2.1 Modal versus amodal perception

Normal image segmentation describes only the visible portion of an object. This is called its **modal** mask. Human vision, however, often represents the whole object, including hidden parts. That complete representation is called the **amodal** object.

For an object of interest, the system is given:

- An input image.
- The object’s visible (modal) mask.
- A semantic category or text prompt, such as “cow,” “motorcycle,” or “giraffe.”

It must produce:

- A completed image showing the object as if it were not occluded.
- An amodal mask covering both visible and newly generated parts.

The object may be hidden by another object, by several objects, or by the image boundary. The completed image can therefore be larger than the original image when the object must be extended beyond an edge.

### 2.2 Why this is difficult

The missing pixels are fundamentally ambiguous. Many different shapes and appearances could be compatible with the visible evidence. Existing approaches commonly use two stages:

1. Predict a complete/amodal binary mask.
2. Generate RGB pixels inside that mask.

The first stage is ill-posed because the correct hidden shape is not uniquely determined. Training data are also difficult to obtain: synthetic occlusions do not perfectly match natural images, while manually labeling hidden object regions is expensive.

A generic diffusion inpainting model does not solve the problem by itself. It may:

- Extend the object too far into regions that should remain background.
- Alter the object’s pose, orientation, or identity.
- Generate a new object that commonly appears with the target object.

For example, when a hand is removed from a cup, the model may generate another hand because cups and hands frequently co-occur in its learned image distribution.

## 3. Main contributions

The paper claims three main contributions:

1. **Progressive Occlusion-aware Completion:** an iterative pipeline that identifies actual occluders and inpaints only those regions, including regions beyond the image boundary.
2. **Mixed Context (MC) Diffusion Sampling:** a training-free method that temporarily breaks the relationship between the target object and its original context, reducing co-occurrence bias and unwanted regenerated occluders.
3. **Counterfactual Completion Curation:** a training-free test for deciding whether a generated object is likely complete by checking how much it grows when its background is outpainted.

## 4. The proposed pipeline

The full system is shown in the paper as a loop. One iteration consists of mask analysis, optional boundary padding, diffusion completion, and an occlusion check. If the object is still occluded, the output becomes the input to another iteration.

### 4.1 Identify the true occluders

The method first applies off-the-shelf grounded instance segmentation (using models such as Segment Anything and Grounding DINO) to obtain masks for objects in the scene.

It then:

1. Finds objects spatially neighboring the query object.
2. Uses an off-the-shelf depth-ordering model to determine which neighboring objects are in front of the query object.
3. Combines those foreground masks into one binary **occlusion mask**.

This is important because the image area outside the query object is not all an occlusion. Most of it is legitimate background and should remain unchanged.

### 4.2 Handle objects touching the image boundary

If the query object touches an image edge, the boundary itself may be hiding part of the object. The method pads the image and the occlusion mask in the relevant directions, allowing the diffusion model to generate pixels outside the original frame.

In the implementation, an object within 10 pixels of the boundary is treated as potentially touching it. The crop used for diffusion is expanded by 60 pixels around the object and by another 60 pixels when boundary extension is needed; the crop is made square before processing.

### 4.3 Crop and inpaint only what is necessary

The method crops around the query object to improve diffusion quality. It then runs a pre-trained Stable Diffusion v2 inpainting model, using:

- The image crop.
- The combined occluder mask.
- A text prompt naming the object category.

Unlike naive outpainting, it does not mask every pixel outside the visible object. This greatly reduces overextension and helps preserve the object’s identity.

### 4.4 Iterate until the object is no longer occluded

After one completion pass, the method reruns segmentation and depth analysis. The generated object may reveal a previously hidden occluder, or it may still touch the padded boundary. If so, another pass is performed.

This progressive strategy addresses a basic limitation of one-shot completion: a single diffusion pass may remove one visible occluder while leaving another hidden behind it. Iteration allows the system to uncover the object layer by layer.

## 5. Mixed Context Diffusion Sampling

### 5.1 The co-occurrence problem

Stable Diffusion has learned statistical relationships between objects and their contexts. Those relationships are usually useful, but they can be harmful for amodal completion. When an occluder is removed, the model may replace it with another object that commonly co-occurs with the query object.

The paper’s solution is not to retrain the model, but to intervene partway through its denoising process.

### 5.2 Two parallel paths

MC sampling creates two related noisy images at an intermediate diffusion timestep (k), then combines them.

#### Path A: complete the query object with a clean context

The original background outside the visible query object is replaced with a simple clean background, similar to a gray product-photography backdrop. The diffusion model then inpaints the occluder region using the object prompt.

Because the original contextual cues have temporarily disappeared, the model is less likely to recreate familiar neighboring objects.

#### Path B: create a clean version of the original background

The query object and its occluders are removed from the original image with a separate removal inpainting model. Noise is added to this object-free background until it reaches the same timestep (k).

This path preserves the original scene while eliminating the objects that could confuse the completion.

### 5.3 Extract an intermediate object mask from diffusion features

At the intermediate timestep, the image is noisy, so ordinary segmentation is unreliable. The authors instead extract internal features from the diffusion model’s U-Net decoder, cluster those features, and use overlap with the known visible mask to select the cluster corresponding to the query object.

The selected cluster becomes an intermediate amodal mask. In experiments, the third decoder layer generally worked best because it represented object geometry and fine visual structure well. The authors typically used timestep 20 out of 50, although the best value depends on occlusion severity.

### 5.4 Composite and finish denoising

The noisy query object from Path A is pasted into the clean, object-free background from Path B using the intermediate amodal mask. The resulting composite is then passed through the remaining diffusion steps.

Conceptually, MC sampling separates two responsibilities:

- Path A supplies a plausible completed object.
- Path B supplies a plausible background without unwanted objects.

The composite reconnects them only after the model has formed a useful object representation.

### 5.5 Why all parts matter

The ablations show that removing parts of MC sampling causes recognizable failures:

- Without removing objects from the background, unwanted co-occurring objects can remain.
- Without replacing the original context, contextual bias is stronger.
- Without feature-based mask extraction, the object and background can be poorly aligned.

## 6. Counterfactual completion curation

The diffusion model can produce several candidate completions, but the system needs a way to decide whether a candidate is complete. The paper proposes a lightweight, training-free heuristic.

### 6.1 Basic intuition

Take the generated object and outpaint its surroundings everywhere except the image corners. Then segment the object again.

- If the original object was already complete, filling in more background should not make the object grow substantially.
- If it was incomplete, the new context may reveal or generate additional missing object parts, causing a large increase in the object mask.

This is a counterfactual test: “What happens to the object if the surrounding scene is changed?”

### 6.2 Decision rule

The rule uses two signals:

1. **Boundary proximity:** if the new object remains very close to an image boundary, it may still be cut off and is judged incomplete.
2. **Mask-area extension:** the new mask is compared with the original amodal mask. A 5×5 dilation allows small changes. If the new mask becomes more than 1.2 times the original area, the object is classified as incomplete.

The thresholds were selected using a validation set of 100 images. They are useful as a practical filter, not as a definitive measure of visual correctness.

## 7. Experimental design

### 7.1 Dataset construction

There is no large natural-image dataset with ground-truth hidden appearances for common object categories. The authors therefore construct a pseudo-occlusion benchmark from COCO and Open Images:

- 3,000 examples in total.
- At least 55 object categories.
- 1,500 easy cases with 20–50% occlusion.
- 1,500 hard cases with 50–80% occlusion.

They simulate an occluded object by placing one complete object over another complete object. The original complete object serves as ground truth for evaluating the generated appearance.

This benchmark provides measurable pixel and feature similarity, but it does not fully capture whether the completed object fits naturally into its original scene. The authors therefore add human preference studies.

### 7.2 Compared methods

The paper compares against:

- **SSSD:** a GAN-based self-supervised scene de-occlusion method.
- **LaMa:** a large-mask image inpainting method.
- **Inst-Inpaint:** a diffusion-based object-removal method.
- **Naive Outpainting:** a baseline that masks everything outside the visible query-object mask.
- **Ours without MC:** the proposed progressive pipeline without Mixed Context sampling.

The main comparisons with prior methods focus on completion inside the original image boundary because the baselines cannot extend beyond it.

### 7.3 Metrics

The authors evaluate at three levels:

- **CLIP cosine similarity:** high-level semantic similarity.
- **DreamSim distance:** mid-level perceptual similarity.
- **LPIPS distance:** lower-level visual similarity.

They also perform human studies asking which completion looks most complete and realistic in context.

## 8. Main results

### 8.1 Quantitative image similarity

The proposed method generally performs best, especially on hard cases. Reported scores are shown as COCO / Open Images.

| Method | Easy CLIP (higher better) | Easy DreamSim (lower better) | Easy LPIPS (lower better) | Hard CLIP (higher better) | Hard DreamSim (lower better) | Hard LPIPS (lower better) | User preference |
|---|---:|---:|---:|---:|---:|---:|---:|
| SSSD | 0.280 / 0.263 | 0.186 / 0.216 | 0.096 / 0.142 | 0.267 / 0.263 | 0.315 / 0.334 | 0.166 / 0.225 | 1.8% |
| LaMa | 0.288 / 0.265 | 0.098 / 0.124 | 0.054 / 0.091 | 0.279 / 0.268 | 0.236 / 0.292 | 0.130 / 0.205 | 7.3% |
| Inst-Inpaint | 0.264 / 0.257 | 0.325 / 0.304 | 0.185 / 0.195 | 0.252 / 0.254 | 0.451 / 0.446 | 0.263 / 0.283 | 0.0% |
| **Ours** | **0.290 / 0.266** | **0.096 / 0.106** | **0.054 / 0.078** | **0.290 / 0.267** | **0.184 / 0.185** | **0.110 / 0.141** | **90.9%** |

The authors caution that LaMa’s numerical scores can look competitive because blurry, similarly colored pixels may be close to the ground truth in low-level metrics. Visual inspection and user preference reveal that such outputs often have weak boundaries and poor object structure.

### 8.2 Ablation results

On 50 easy and 50 hard natural-image cases:

| Method | Easy successful completions | Easy user preference | Hard successful completions | Hard user preference |
|---|---:|---:|---:|---:|
| Naive Outpainting | 66% | 18% | 40% | 18% |
| Ours without MC | 90% | 36% | 72% | 28% |
| **Ours** | **88%** | **48%** | **90%** | **54%** |

The most important pattern is the hard-case result: MC sampling raises successful completion from 72% to 90%, an 18-point improvement, when the occluder is a common co-occurring object. The small drop from 90% to 88% on easy cases suggests that MC is mainly valuable when contextual bias is severe.

### 8.3 MC design choices

The authors report three useful findings:

- A gray clean background increased successful completions by about 20% relative to retaining the original context.
- Forest and sky backgrounds performed worse than the original context, by about 16% and 9%, respectively. This suggests that the clean background should be neutral rather than semantically suggestive.
- Swapping contexts around diffusion timestep 20 of 50 produced the best results on the tested difficult cases.

### 8.4 Completeness classifier

On a test set of 50 complete and 50 incomplete objects:

| Classifier | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Human consensus | 0.83 | 0.75 | 0.99 |
| Proposed rule | 0.70 | 0.68 | 0.68 |
| Random chance | 0.50 | 0.50 | 0.50 |

The rule is clearly better than chance, but substantially worse than humans. The authors present it as a useful filter and starting point, not a solved evaluation problem.

## 9. Interpretation of the results

The experiments support four claims:

1. **Selective masking matters.** Inpainting only detected occluders preserves background and object identity better than masking all non-object pixels.
2. **Iteration matters.** Progressive passes can uncover multiple layers of occlusion that a one-shot method misses.
3. **Context can be harmful.** The original scene provides helpful constraints, but it also activates learned co-occurrence patterns. Temporarily removing that context makes the diffusion model more willing to complete the target object itself.
4. **Pixel metrics are incomplete.** A blurry result may score well on low-level similarity while looking obviously wrong to people. Human evaluation is especially important for contextual plausibility.

## 10. Strengths of the approach

### Training-free and practical

The method reuses a public Stable Diffusion inpainting checkpoint and off-the-shelf segmentation and depth models. It avoids collecting amodal masks and avoids expensive retraining.

### Handles several occlusion types

It can address:

- Objects hidden by another object.
- Multiple occluders.
- Objects cut off by the image boundary.
- Combinations of these cases.

### Produces diverse hypotheses

Because the core generator is diffusion-based, multiple plausible completions can be sampled rather than producing one deterministic answer.

### Better separation of object and background

MC sampling explicitly treats the completed object and its context as different generation problems before recombining them. This is a useful design principle for other image-editing tasks where context creates unwanted biases.

### Potential downstream uses

The authors demonstrate or suggest applications in:

- Robotics and autonomous systems, where hidden object geometry matters.
- Augmented reality and image editing.
- Dense visual correspondence.
- Novel-view or 3D object synthesis from a single image.

## 11. Limitations and failure cases

The paper is candid that amodal completion remains difficult.

### Occluder completely encloses the object

If a small query object is surrounded by a larger occluder, the visible evidence may be too weak. The method can overextend the object or hallucinate an implausible shape.

### Shadows and contextual traces

Shadows associated with the query object or occluder can act as contextual cues. The diffusion model may interpret them as evidence for another compatible object and generate an unwanted occluder.

### Human-object interactions and pose

Certain poses strongly imply interaction, such as a person riding a horse. The model may generate a related object even when the desired completion should isolate the person, or it may struggle to infer hidden limbs and body geometry.

### Dependence on auxiliary models

Errors in grounded segmentation, depth ordering, or feature clustering can propagate through the pipeline. If an occluder is missed, the diffusion model receives the wrong mask; if an object mask is wrong, the completion can be distorted.

### Heuristic completeness test

The counterfactual rule is simple and training-free, but its 70% accuracy is not sufficient for fully automatic quality control. A learned curator trained on a carefully labeled set could likely improve it.

### Evaluation limitations

The benchmark uses synthetic pseudo-occlusions made from natural images. This is useful because it provides ground truth, but it does not reproduce every property of real occlusion, including lighting, contact shadows, physical interactions, and correlated camera viewpoints.

### Computational cost

The system chains several large models and may run multiple diffusion iterations. The experiments use a 24 GB NVIDIA Titan RTX, so deployment on constrained hardware may require optimization.

## 12. Key technical details from the appendix

- Base generator: publicly available Stable Diffusion v2 inpainting model.
- Diffusion schedule: 50 DDIM steps in the main setup.
- Typical MC feature layer: third U-Net decoder layer.
- Typical MC intervention point: timestep 20 of 50.
- Boundary proximity threshold for identifying possible truncation: 10 pixels.
- Counterfactual boundary threshold: 2 pixels.
- Counterfactual mask-area threshold: more than 1.2 times the previous amodal mask indicates likely incompleteness.
- The intermediate feature map is flattened, clustered with unsupervised least-squares quantization, then aligned to the visible object mask.
- User studies use attention checks; workers must pass the checks to have their labels included.

## 13. Plain-language walkthrough with an example

Imagine a photograph of a cow partly hidden behind a fence and touching the right edge of the image.

1. The system segments the cow and nearby objects.
2. Depth analysis identifies the fence as being in front of the cow.
3. Because the cow touches the right edge, the image is padded on that side.
4. The system masks only the fence and boundary region, not the entire background.
5. MC sampling temporarily places the visible cow on a neutral background and separately creates a cow-free version of the original scene.
6. At an intermediate noise level, internal diffusion features identify the likely full cow shape.
7. The completed cow is composited onto the clean scene background and denoising continues.
8. The result is checked for remaining occluders. If another object is still in front, another iteration runs.
9. Finally, the system outpaints the background as a counterfactual test. If the cow mask expands dramatically, the completion is flagged as incomplete; otherwise, it is accepted.

## 14. Overall assessment

This paper’s main contribution is a clever way to turn a general-purpose diffusion model into an amodal-completion system without training it specifically for that task. The method recognizes that two problems must be solved simultaneously:

- The model must invent hidden object pixels.
- The model must avoid inventing the wrong objects because of scene context.

Progressive occlusion analysis addresses the first problem by controlling *where* generation occurs. Mixed Context sampling addresses the second by controlling *which context* the model sees during the crucial part of denoising. The counterfactual curator provides a practical, if imperfect, way to reject incomplete results.

The results are convincing for many natural-image cases, especially high-occlusion and co-occurrence-heavy examples. However, the method is still best understood as plausible visual reconstruction rather than guaranteed recovery of the true hidden object. When the visible evidence is weak or strongly tied to another object, the system can produce a realistic-looking but incorrect completion.

## 15. Takeaways

- Amodal completion means reconstructing an object’s hidden and out-of-frame parts, not merely filling a hole in the background.
- A generic outpainting call is too unconstrained and often changes the object or creates unwanted co-occurring objects.
- Detecting actual occluders and completing them progressively is more reliable than masking everything around the object.
- Mixed Context Diffusion Sampling reduces contextual co-occurrence bias without training or fine-tuning.
- A counterfactual outpainting test can filter some incomplete generations, but human judgments remain better.
- The method is a strong proof of concept for training-free object completion, with remaining challenges in severe enclosure, shadows, interactions, and ambiguous poses.

