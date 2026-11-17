package aitest

type ModelOption interface{ applyModel(*Model) }

type BatchOption interface{ applyBatch(*BatchModel) }

type EmbeddingOption interface{ applyEmbedding(*EmbeddingModel) }

type RerankOption interface{ applyRerank(*RerankModel) }

type ImageOption interface{ applyImage(*ImageModel) }

type VideoOption interface{ applyVideo(*VideoModel) }

type SpeechOption interface{ applySpeech(*SpeechModel) }

type TranscriptionOption interface{ applyTranscription(*TranscriptionModel) }

type RealtimeOption interface{ applyRealtime(*RealtimeModel) }

// FailOption makes the first len(errs) calls of a model's primary operation fail with the given errors.
type FailOption struct{ errs []error }

// Fail returns an option accepted by every New...Model constructor. The failing operation is Generate or Stream for NewModel, StartBatch, Embed, Rerank, GenerateImage, StartVideo, GenerateSpeech, Transcribe and Connect for the others.
func Fail(errs ...error) FailOption { return FailOption{errs: errs} }

func (o FailOption) applyModel(m *Model)                      { m.failures = o.failures() }
func (o FailOption) applyBatch(m *BatchModel)                 { m.failures = o.failures() }
func (o FailOption) applyEmbedding(m *EmbeddingModel)         { m.failures = o.failures() }
func (o FailOption) applyRerank(m *RerankModel)               { m.failures = o.failures() }
func (o FailOption) applyImage(m *ImageModel)                 { m.failures = o.failures() }
func (o FailOption) applyVideo(m *VideoModel)                 { m.failures = o.failures() }
func (o FailOption) applySpeech(m *SpeechModel)               { m.failures = o.failures() }
func (o FailOption) applyTranscription(m *TranscriptionModel) { m.failures = o.failures() }
func (o FailOption) applyRealtime(m *RealtimeModel)           { m.failures = o.failures() }

func (o FailOption) failures() failures { return failures{errs: o.errs} }

type failures struct {
	errs []error
	used int
}

func (f *failures) next() error {
	if f.used >= len(f.errs) {
		return nil
	}
	f.used++
	return f.errs[f.used-1]
}
