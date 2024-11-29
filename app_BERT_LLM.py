import streamlit as st
from bias_model_BERT import BiasDetector
from unbias_model_LLM import LLMTextDebiaser
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

class EnhancedBiasAnalyzerApp:
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        try:
            self.bias_detector = BiasDetector(model_path="models/bias_bert_model_synthetic")
            self.debiaser = LLMTextDebiaser()
            self.logger.info("Models initialized successfully")
        except Exception as e:
            st.error(f"Error initializing models: {e}")
            return

    def get_token_probabilities(self, text: str):
        """Get token-wise probabilities for the input text"""
        tokenizer = self.bias_detector.tokenizer
        model = self.bias_detector.model
        device = self.bias_detector.device

        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # Get attention weights from the model
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            # Get attention weights from the last layer
            attention = outputs.attentions[-1]  # Shape: [batch, num_heads, seq_len, seq_len]
            # Average across attention heads
            avg_attention = attention.mean(dim=1)  # Shape: [batch, seq_len, seq_len]
            # Get attention scores for [CLS] token
            token_importances = avg_attention[0, 0, :].cpu()

            # Get overall bias probability
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            overall_bias_prob = probs[0, 1].item()  # Probability of bias class

        # Get token-wise information
        token_probs = []
        input_ids = inputs['input_ids'][0]
        
        for i, token_id in enumerate(input_ids):
            if i >= len(token_importances):
                break
            
            token = tokenizer.decode(token_id)
            # Normalize attention score to [0,1] range and combine with overall bias probability
            attention_score = token_importances[i].item()
            # Scale attention score by overall bias probability
            scaled_prob = attention_score * overall_bias_prob
            
            token_probs.append({
                'token': token,
                'probability': scaled_prob,
                'attention_score': attention_score,
                'is_bias': scaled_prob > 0.5
            })

        return token_probs

    def visualize_token_probabilities(self, token_probs):
        """Create a visualization of token probabilities"""
        tokens = [tp['token'].replace('Ġ', ' ').strip() for tp in token_probs]
        probs = [tp['probability'] for tp in token_probs]
        attention_scores = [tp['attention_score'] for tp in token_probs]
        
        # Create two traces: one for probabilities and one for attention scores
        fig = go.Figure(data=[
            go.Bar(
                name='Bias Probability',
                x=tokens,
                y=probs,
                marker_color=['rgba(255,0,0,0.7)' if p > 0.5 else 'rgba(0,255,0,0.7)' for p in probs],
                text=[f'{p:.3f}' for p in probs],
                textposition='auto',
            ),
            go.Bar(
                name='Attention Score',
                x=tokens,
                y=attention_scores,
                marker_color='rgba(0,0,255,0.3)',
                text=[f'{a:.3f}' for a in attention_scores],
                textposition='auto',
                visible='legendonly'
            )
        ])

        fig.update_layout(
            title='Token-wise Analysis',
            xaxis_title='Tokens',
            yaxis_title='Score',
            yaxis_range=[0, 1],
            height=400,
            barmode='overlay',
            showlegend=True
        )

        return fig

    def display_token_highlights(self, text: str, token_probs):
        """Create HTML with highlighted tokens based on probabilities"""
        html = ""
        current_pos = 0
        
        for tp in token_probs:
            token = tp['token'].replace('Ġ', ' ').replace('Ċ', '\n')
            prob = tp['probability']
            
            # Calculate color intensity based on probability
            red = int(255 * min(prob * 2, 1))  # Amplify the effect for visibility
            green = int(255 * (1 - min(prob * 2, 1)))
            
            # Add the highlighted token
            if token.startswith(' '):
                html += f' <span style="background-color: rgba({red},{green},0,0.3);" title="Bias prob: {prob:.3f}">{token[1:]}</span>'
            else:
                html += f'<span style="background-color: rgba({red},{green},0,0.3);" title="Bias prob: {prob:.3f}">{token}</span>'

        return html

    def run(self):
        st.title("Text Bias Analyzer")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Enter your text here..."
        )
        
        if st.button("Analyze Text"):
            if text_input.strip():
                # Add spinner during processing
                with st.spinner("Analyzing text..."):
                    # Get token probabilities
                    token_probs = self.get_token_probabilities(text_input)
                    
                    # Detect bias
                    analysis = self.bias_detector.analyze_text(text_input)
                    
                    # Display results
                    st.header("Analysis Results")
                    
                    # Create columns for metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Overall Bias Detection", 
                            analysis.prediction,
                            f"{analysis.confidence*100:.1f}% confidence"
                        )
                    
                    # Display token probability visualization
                    st.subheader("Token-wise Analysis")
                    fig = self.visualize_token_probabilities(token_probs)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display text with token highlighting
                    st.markdown("### Text with Token-level Bias Analysis")
                    highlighted_text = self.display_token_highlights(text_input, token_probs)
                    st.markdown(
                        f'<div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd;">{highlighted_text}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # If text is biased, show details and generate unbiased version
                    if analysis.is_biased:
                        # Show biased phrases
                        if analysis.biased_phrases:
                            st.markdown("### Potentially Biased Phrases")
                            for phrase in analysis.biased_phrases:
                                st.markdown(f"- {phrase}")
                        
                        # Generate and show debiased version
                        st.header("Debiased Version")
                        with st.spinner("Generating unbiased version..."):
                            debiased_text = self.debiaser.debias_text(
                                text_input,
                                analysis.biased_phrases
                            )
                        
                        # Display original and debiased versions side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Original Text")
                            st.write(text_input)
                        with col2:
                            st.subheader("Debiased Text")
                            st.write(debiased_text)
                    
                    else:
                        st.success("This text appears to be unbiased!")
            else:
                st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    app = EnhancedBiasAnalyzerApp()
    app.run()