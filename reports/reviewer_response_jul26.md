# Response to reviewer comments (July 2026)

Draft email to the reviewer. No em dashes; formal register.

---

**Subject:** Revised final report: adjustments for public release

Dear [Reviewer],

Thank you for the further comments. You are right that the document comparison was of little help: the revision reworked a large share of the text rather than editing it line by line, so Word could not align the two versions reliably. With that in mind, we have made the adjustments you asked for so that the published report stands on its own.

The main change concerns the references to the earlier version. Because only the final report will be published, we have removed every reference to the original report throughout the document. This includes the cover page, the roughly two dozen in-text mentions, the dedicated section that related the two versions, and the appendix that listed the changes between them. We have also removed the discussion of the measurement and logging issues encountered along the way. Those were resolved before the final results were produced, and a reader of the published report is served better by a clean description of the final state than by an account of how we got there.

We have also addressed the smaller points:

- The lone third-level heading (4.4.1) has been removed, with its content merged into its parent section, so there is no longer an only-child subsection.
- The cover page and text now refer to the "IEA R&D Fund Call 4-5" rather than to an "R&D Committee."

Two of your comments raise questions that we would like to answer directly.

**On a public-facing interface along the lines of "Ask NAEP."** We would lean towards recommending it. Making IEA's public outputs easier to discover through a natural-language interface is a legitimate and valuable use, distinct from the internal, privacy-preserving use case that motivated this phase, and the retrieval results here suggest it is technically within reach. We would attach one condition before any such service is offered: a human-expert evaluation to confirm that the answer quality is high enough for a public audience, since a public tool carries more reputational risk than an internal one. The examples you cite are a useful reference point for scope and framing.

**On the value of a local RAG system compared with a managed service such as Claude that can ingest documents in a safe environment.** The advantage of retrieval grows with the size of the source collection. A managed service can answer accurately when a small number of documents are attached, because their full text fits within the model's context window. IEA's documentation does not: across studies it runs to thousands of documents and well beyond any single model's context window, so the material has to be indexed and retrieved rather than read in full for each question. That is precisely what a RAG system does, and it is the regime in which it is needed. Alongside this, a local deployment keeps sensitive or embargoed material on IEA's own hardware, avoids per-seat or per-token costs as usage scales, and makes every answer traceable to the source passages behind it.

The revised report and its appendices are attached. We are happy to make further adjustments if anything remains unclear.

With kind regards,

Widianto Persadha, on behalf of the authors
