"""OpenAI-powered agents for extracting and formatting investment update information."""
import json
import time
import openai
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, EXTRACTION_MODEL, FORMATTING_MODEL
from benchmark_lookup import BenchmarkLookup
from text_chunker import estimate_tokens, chunk_text


# System prompt for the extraction agent
# EXTRACTION_PROMPT = """You are a highly experienced investment specialist. You are reviewing investment updates that come in a wide array of presentations and formats, in an effort to distill key performance metrics, both qualitative and quantitative. Your task is to extract critical information from investment update documents and return it as structured JSON.

# CRITICAL: Always extract fund-level Net IRR, Net MOIC (or TVPI), and Net DPI as separate numeric fields. These are essential for benchmarking.

# IMPORTANT - When multiple values exist (e.g., fund-level vs. investor-level):
# - ALWAYS prioritize values explicitly labeled as "fund level", "fund-level", or "fund level performance"
# - NEVER use values labeled for specific investors (e.g., "for a $500K advisory investor", "for advisory investors", "investor-level")
# - If you see both fund-level and investor-specific values, extract ONLY the fund-level value
# - If only investor-specific values are present, extract those but note the limitation

# For each investment update document provided, extract the following information and return it as a JSON object with these exact keys:

# {
#   "fund_name": "The exact name of the fund or company",
#   "asset_class": "The asset class (e.g., 'Private Equity', 'Venture Capital', 'Real Estate', 'Credit', 'Private Debt', 'Hedge Fund', 'Infrastructure', 'Natural Resources', etc.)",
#   "deal_type": "A classification of the investment following menu: [Fund, Direct Investment]"
#   "vintage": "The vintage year as a 4-digit number (e.g., 2020, 2021) or null if not found",
#   "net_irr": "Net IRR as a number (e.g., 15.5 for 15.5%) or null if not found",
#   "net_moic": "Net MOIC or TVPI as a number (e.g., 2.5 for 2.5x) or null if not found",
#   "net_dpi": "Net DPI as a number (e.g., 1.2 for 1.2x) or null if not found",
#   "performance_summary": "A 1-2 word performance summary from the following menu: [As Expected, Outperforming, Underperforming] - this will be determined after benchmark comparison",
#   "investment_performance": [
#     "Key metric, return, or performance data point 1",
#     "Key metric, return, or performance data point 2",
#     "includes quantitative performance metrics, revenue growth, returns, margin expansion,financial data, and new developments",
#     "... (as many items as necessary)"
#   ],
#   "key_takeaways": [
#     "Financial performance metric 1",
#     "Benchmark comparison 1",
#     "Quantitative measure 1",
#     "Qualitative measure 1",
#     "... (as many items as necessary)"
#   ],
#   "business_updates": [
#     "Strategic update or business development 1",
#     "Market condition or commentary 1",
#     "... (as many items as necessary)"
#   ]
# }

# IMPORTANT:
# - You must not hallucinate under any circumstances. If you are not sure about a piece of information, return null. If you hallucinate, or get any of the information incorrect, someone will die and you will be held responsible. 
# - Always extract Asset Class and Vintage explicitly. If not clearly stated, use your best judgment based on context, or use null/empty string if truly unavailable.
# - Extract Net IRR, Net MOIC/TVPI, and Net DPI as numeric values (remove % signs, keep as numbers). If not found, use null.
# - CRITICAL: For Net IRR, Net MOIC, and Net DPI - if the document shows both fund-level and investor-specific values, you MUST extract the fund-level value. Look for phrases like "fund level", "fund-level", "at the fund level" and ignore values labeled "for a $X investor", "advisory investor", "investor-level", etc.
# - PRIORITIZE performance metrics, returns, financial data, and new developments over generic overview information.
# - EXCLUDE generic statements like "this fund focuses on X sector" or "the fund invests in Y" - these are not helpful.
# - FOCUS on: performance numbers, returns, exits, new investments, portfolio company updates, market conditions, strategic changes.
# - Return ONLY valid JSON, no additional text or markdown formatting.
# - Use arrays for the three detail sections (investment_performance, key_takeaways, business_updates) - each item should be a clear, concise statement.
# - Be thorough but selective - capture important performance and news, skip generic descriptions."""


# # System prompt for the formatting agent
# FORMATTING_PROMPT = """You are an investment update formatter. Your task is to format extracted investment information (provided as JSON) into a standardized text format.

# You will receive a JSON object with the following structure:
# - fund_name: The fund/company name
# - asset_class: The asset class
# - deal_type: The deal type (Fund or Direct Investment)
# - vintage: The vintage year (or null)
# - performance_summary: Brief performance summary (As Expected, Outperforming, or Underperforming)
# - investment_performance: Array of investment performance details (including benchmark comparisons)
# - key_takeaways: Array of key takeaways
# - business_updates: Array of business updates and market commentary

# Format the information as follows:

# {Fund/Company Name} Update - {Performance Summary}

# **Quantitative Performance:**
#   • Net IRR: [net_irr]%[ (vs benchmark [benchmark_irr]% - [benchmark_category]) if benchmark_irr_comparison exists]
#   • Net MOIC: *OMIT IF DIRECT INVESTMENT* [net_moic]x[ (vs benchmark [benchmark_moic]x - [benchmark_category]) if benchmark_moic_comparison exists]
#   • Net DPI: *OMIT IF DIRECT INVESTMENT* [net_dpi]x[ (vs benchmark [benchmark_dpi]x - [benchmark_category]) if benchmark_dpi_comparison exists]
#   • [First bullet point from investment_performance array]
#   • [Second bullet point from investment_performance array]
#   • [Continue for all items...]

# **Key Takeaways and Business Updates:**
#   • [First bullet point from key_takeaways array]
#   • [Second bullet point from key_takeaways array]
#   • [Continue for all items...]

# **Market Commentary:**
#   • [First bullet point from business_updates array]
#   • [Second bullet point from business_updates array]
#   • [Continue for all items...]

# Guidelines:
# - Your job is to deduce what is most essential, even if it is only one sentence. Quality is more important than verbosity. The update should not exceed 250 words.
# - Use the fund_name and performance_summary from the JSON for the header
# - For IRR/MOIC/DPI lines: If benchmark_comparison data exists (benchmark_irr_comparison, benchmark_moic_comparison, benchmark_dpi_comparison), include the benchmark comparison inline in parentheses. Format: "Net IRR: 15.5% (vs benchmark 13.1% - Above Median)" or similar.
# - Only include benchmark comparisons if they are present in the JSON (they will only be present if asset_class and vintage are available)
# - Section titles (Quantitative Performance:, Key Takeaways and Business Updates:, Market Commentary:) MUST be bolded using **text** markdown syntax
# - Use the bullet character (•) for all bullet points - this will be automatically converted to proper Google Docs bullet formatting
# - Use indented bullet points with two spaces of indentation before each bullet: "  • "
# - Convert each array item into a bullet point
# - Use clear, concise bullet points
# - CRITICAL: Each section (Investment Performance, Key Takeaways, Business Updates/Market Commentary) must NOT exceed 200 words total
# - PRIORITIZE: Performance metrics, returns, financial data, new developments, portfolio company news
# - EXCLUDE: Generic overview statements like "this fund focuses on X" or "the fund invests in Y"
# - If a section would exceed 200 words, prioritize the most important/insightful items and remove less critical ones
# - Ensure each section has at least one bullet point
# - Use professional language
# - If an array is empty or has no relevant information, include a bullet like "  • No significant updates"
# - The Performance Summary in the header should be brief (e.g., "Strong Q4 Performance", "Challenging Market Conditions", "Up 15% YoY")
# - Make sure every bullet point starts with "  • " (two spaces, bullet character •, space)
# - Focus on actionable insights and quantitative data over descriptive overviews"""
EXTRACTION_PROMPT = """You are a highly experienced investment specialist. You are reviewing investment updates that come in a wide array of presentations and formats, in an effort to distill key performance metrics, both qualitative and quantitative. Your task is to extract critical information from investment update documents and return it as structured JSON.

CRITICAL: Always extract fund-level Net IRR, Net MOIC (or TVPI), and Net DPI as separate numeric fields. These are essential for benchmarking.

DOCUMENT TYPES WITH NO FUND-LEVEL METRICS (set net_irr, net_moic, net_dpi to null):
- Investor presentations, pitch decks, investment memos, or announcements (e.g. trip, final close, calendar) that do NOT report fund-level returns.
- If the document clearly has no fund-level performance figures (e.g. "No performance metrics to show", or presentation-only with no fund IRR/MOIC/DPI), set net_irr, net_moic, and net_dpi to null. Do NOT use portfolio-company returns, deal-level IRR/MOIC, or company financials (revenue, EBITDA multiples) as fund-level metrics.

IMPORTANT - When multiple values exist (e.g., fund-level vs. investor-level):
- ALWAYS prioritize values explicitly labeled as "fund level", "fund-level", or "fund level performance"
- NEVER use values labeled for specific investors (e.g., "for a $500K advisory investor", "for advisory investors", "investor-level")
- If you see both fund-level and investor-specific values, extract ONLY the fund-level value
- If only investor-specific values are present, extract those but note the limitation
- ONLY populate net_irr, net_moic, net_dpi when values are explicitly fund-level (or equivalent). Do NOT use deal-level or portfolio-company IRR/MOIC/DPI as fund-level.

For each investment update document provided, extract the following information and return it as a JSON object with these exact keys:

{
  "fund_name": "The exact name of the fund or company",
  "asset_class": "The asset class (e.g., 'Private Equity', 'Venture Capital', 'Real Estate', 'Credit', 'Private Debt', 'Hedge Fund', 'Infrastructure', 'Natural Resources', etc.)",
  "deal_type": "A classification of the investment following menu: [Fund, Direct Investment]"
  "vintage": "The vintage year as a 4-digit number (e.g., 2020, 2021) or null if not found",
  "net_irr": "Net IRR as a number (e.g., 15.5 for 15.5%) or null if not found",
  "net_moic": "Net MOIC or TVPI as a number (e.g., 2.5 for 2.5x) or null if not found",
  "net_dpi": "Net DPI as a number (e.g., 1.2 for 1.2x) or null if not found",
  "performance_summary": "A 1-2 word performance summary from the following menu: [As Expected, Outperforming, Underperforming] - this will be determined after benchmark comparison",
  "investment_performance": [
    "Key metric, return, or performance data point 1",
    "Key metric, return, or performance data point 2",
    "includes quantitative performance metrics, revenue growth, returns, margin expansion,financial data, and new developments",
    "... (as many items as necessary)"
  ],
  "key_takeaways": [
    "Financial performance metric 1",
    "Benchmark comparison 1",
    "Quantitative measure 1",
    "Qualitative measure 1",
    "... (as many items as necessary)"
  ],
  "business_updates": [
    "Strategic update or business development 1",
    "Market condition or commentary 1",
    "... (as many items as necessary)"
  ]
}

IMPORTANT:
- You must not hallucinate under any circumstances. If you are not sure about a piece of information, return null. If you hallucinate, or get any of the information incorrect, someone will die and you will be held responsible. 
- Always extract Asset Class and Vintage explicitly. If not clearly stated, use your best judgment based on context, or use null/empty string if truly unavailable.
- Extract Net IRR, Net MOIC/TVPI, and Net DPI as numeric values (remove % signs, keep as numbers). If not found, use null.
- CRITICAL: For Net IRR, Net MOIC, and Net DPI - if the document shows both fund-level and investor-specific values, you MUST extract the fund-level value. Look for phrases like "fund level", "fund-level", "at the fund level" and ignore values labeled "for a $X investor", "advisory investor", "investor-level", etc.
- PRIORITIZE performance metrics, returns, financial data, and new developments over generic overview information.
- EXCLUDE generic statements like "this fund focuses on X sector" or "the fund invests in Y" - these are not helpful.
- FOCUS on: performance numbers, returns, exits, new investments, portfolio company updates, market conditions, strategic changes.
- Return ONLY valid JSON, no additional text or markdown formatting. Output a single, well-formed JSON object (no trailing prose, no invalid escape sequences, no embedded snippets).
- Use arrays for the three detail sections (investment_performance, key_takeaways, business_updates) - each item should be a clear, concise statement.
- Be thorough but selective - capture important performance and news, skip generic descriptions."""

class ExtractionAgent:
    """Agent responsible for extracting structured information from investment update text."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.benchmark_lookup = BenchmarkLookup()
    
    def extract_information(self, text, investment_name=None):
        """
        Extract structured information from investment update text.
        
        Args:
            text: The text content from the PDF
            investment_name: Optional name of the investment (for fallback)
            
        Returns:
            Tuple of (extracted_data_dict, metadata_dict, metrics_dict)
        """
        start_time = time.time()
        
        # Estimate tokens and check if chunking is needed
        prompt_overhead = estimate_tokens(EXTRACTION_PROMPT, EXTRACTION_MODEL) + 100
        max_tokens = 6000  # Leave room for response
        
        text_tokens = estimate_tokens(text, EXTRACTION_MODEL)
        
        if text_tokens + prompt_overhead <= max_tokens:
            # Single extraction
            extracted_data, token_info = self._extract_single(text, investment_name)
        else:
            # Chunked extraction
            extracted_data, token_info = self._extract_from_chunks(text, investment_name)
        
        latency_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Calculate cost for extraction agent (gpt-5.2)
        # Pricing: Input $1.75/1M tokens, Output $14.00/1M tokens
        prompt_tokens = token_info.get('prompt_tokens', 0)
        completion_tokens = token_info.get('completion_tokens', 0)
        extraction_cost = (prompt_tokens / 1_000_000 * 1.75) + (completion_tokens / 1_000_000 * 14.00)
        
        # Extract metadata
        metadata = {
            'fund_name': extracted_data.get('fund_name') or investment_name or 'Unknown',
            'asset_class': extracted_data.get('asset_class') or 'Unknown',
            'vintage': extracted_data.get('vintage') or None,
            'performance_summary': extracted_data.get('performance_summary') or 'Unknown'
        }
        
        metrics = {
            'extraction_tokens': token_info.get('total_tokens', 0),
            'extraction_prompt_tokens': prompt_tokens,
            'extraction_completion_tokens': completion_tokens,
            'extraction_cost': extraction_cost,
            'extraction_latency_ms': latency_ms
        }
        
        return extracted_data, metadata, metrics
    
    def _extract_single(self, text, investment_name=None):
        """Extract information from a single text chunk. Returns (extracted_data, token_info_dict)."""
        try:
            user_message = f"Extract information from the following investment update:\n\n{text}"
            
            response = self.client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=OPENAI_TEMPERATURE
            )
            
            # Track token usage
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                total_tokens = response.usage.total_tokens or 0
            
            extracted_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                extracted_data = json.loads(extracted_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if '```json' in extracted_text:
                    json_start = extracted_text.find('```json') + 7
                    json_end = extracted_text.find('```', json_start)
                    extracted_text = extracted_text[json_start:json_end].strip()
                    extracted_data = json.loads(extracted_text)
                elif '```' in extracted_text:
                    json_start = extracted_text.find('```') + 3
                    json_end = extracted_text.find('```', json_start)
                    extracted_text = extracted_text[json_start:json_end].strip()
                    extracted_data = json.loads(extracted_text)
                else:
                    raise
            
            # Store original performance summary before benchmark comparison
            original_performance_summary = extracted_data.get('performance_summary')
            
            # Add benchmark comparisons if asset class and vintage are available
            extracted_data = self._add_benchmark_comparisons(extracted_data)
            
            # Determine performance summary based on benchmark comparisons (falls back to original if benchmarks unavailable)
            extracted_data['performance_summary'] = self._determine_performance_summary(extracted_data, original_performance_summary)
            
            token_info = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens
            }
            
            return extracted_data, token_info
        except Exception as e:
            print(f"Error extracting information with OpenAI: {e}")
            raise
    
    def _extract_from_chunks(self, text, investment_name=None):
        """Extract information from multiple text chunks and merge results. Returns (extracted_data, token_info_dict)."""
        chunks = chunk_text(text, max_tokens=6000, model=EXTRACTION_MODEL, overlap=200)
        extractions = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            try:
                extracted, token_info = self._extract_single(chunk, investment_name)
                extractions.append(extracted)
                total_prompt_tokens += token_info.get('prompt_tokens', 0)
                total_completion_tokens += token_info.get('completion_tokens', 0)
                total_tokens += token_info.get('total_tokens', 0)
            except Exception as e:
                print(f"Error extracting from chunk {i+1}: {e}")
                continue
        
        if not extractions:
            raise Exception("Failed to extract information from any chunk")
        
        # Merge extractions
        merged_data = self._merge_extractions(extractions, investment_name)
        token_info = {
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'total_tokens': total_tokens
        }
        return merged_data, token_info
    
    def _merge_extractions(self, extractions, investment_name=None):
        """Merge multiple extraction results into a single result."""
        if not extractions:
            return {}
        
        # Start with the first extraction
        merged = extractions[0].copy()
        
        # Merge arrays (deduplicate and combine)
        for key in ['investment_performance', 'key_takeaways', 'business_updates']:
            if key not in merged:
                merged[key] = []
            
            seen = set(merged[key])
            for extraction in extractions[1:]:
                if key in extraction:
                    for item in extraction[key]:
                        if item not in seen:
                            merged[key].append(item)
                            seen.add(item)
        
        # For scalar fields, prefer non-null values
        # For numeric metrics (IRR, MOIC, DPI), prefer higher values as they're more likely to be fund-level
        for key in ['fund_name', 'asset_class', 'vintage', 'performance_summary']:
            for extraction in extractions:
                if key in extraction and extraction[key] is not None and (not merged.get(key) or merged[key] == 'Unknown' or merged[key] is None):
                    merged[key] = extraction[key]
        
        # For numeric metrics, prefer higher values (fund-level typically higher than investor-level due to fees)
        for key in ['net_irr', 'net_moic', 'net_dpi']:
            for extraction in extractions:
                if key in extraction and extraction[key] is not None:
                    current_value = merged.get(key)
                    if current_value is None or current_value == 'Unknown':
                        merged[key] = extraction[key]
                    elif isinstance(extraction[key], (int, float)) and isinstance(current_value, (int, float)):
                        # Prefer higher value (more likely to be fund-level)
                        merged[key] = max(current_value, extraction[key])
                    else:
                        # If types don't match, keep current
                        pass
        
        # Store original performance summary before benchmark comparison
        original_performance_summary = merged.get('performance_summary')
        
        # Add benchmark comparisons after merging
        merged = self._add_benchmark_comparisons(merged)
        
        # Determine performance summary based on benchmark comparisons (falls back to original if benchmarks unavailable)
        merged['performance_summary'] = self._determine_performance_summary(merged, original_performance_summary)
        
        return merged
    
    def _add_benchmark_comparisons(self, extracted_data):
        """Add benchmark comparison data to extracted_data for inline formatting."""
        asset_class = extracted_data.get('asset_class')
        vintage = extracted_data.get('vintage')
        
        # Only calculate benchmarks if asset_class and vintage are available
        if not self.benchmark_lookup or not asset_class or not vintage:
            return extracted_data
        
        # Compare IRR
        net_irr = extracted_data.get('net_irr')
        if net_irr is not None:
            irr_comparison = self.benchmark_lookup.compare_irr(asset_class, vintage, net_irr)
            if irr_comparison:
                extracted_data['benchmark_irr_comparison'] = {
                    'median': irr_comparison['median'],
                    'category': irr_comparison['category'],
                    'percentile': irr_comparison['percentile']
                }
        
        # Compare MOIC
        net_moic = extracted_data.get('net_moic')
        if net_moic is not None:
            moic_comparison = self.benchmark_lookup.compare_moic(asset_class, vintage, net_moic)
            if moic_comparison:
                extracted_data['benchmark_moic_comparison'] = {
                    'median': moic_comparison['median'],
                    'category': moic_comparison['category'],
                    'percentile': moic_comparison['percentile']
                }
        
        # Compare DPI
        net_dpi = extracted_data.get('net_dpi')
        if net_dpi is not None:
            dpi_comparison = self.benchmark_lookup.compare_dpi(asset_class, vintage, net_dpi)
            if dpi_comparison:
                extracted_data['benchmark_dpi_comparison'] = {
                    'median': dpi_comparison['median'],
                    'category': dpi_comparison['category'],
                    'percentile': dpi_comparison['percentile']
                }
        
        return extracted_data
    
    def _determine_performance_summary(self, extracted_data, original_performance_summary=None):
        """
        Determine performance summary based on benchmark comparisons.
        Falls back to combined quantitative and qualitative assessment if benchmarks unavailable.
        
        Args:
            extracted_data: The extracted data dictionary
            original_performance_summary: The performance summary extracted from the document
            
        Returns:
            Performance summary: 'Outperforming', 'Underperforming', or 'As Expected'
        """
        asset_class = extracted_data.get('asset_class')
        vintage = extracted_data.get('vintage')
        
        # Collect all performance percentiles from benchmark comparisons
        # Use the comparison data already stored in extracted_data if available
        percentiles = []
        
        if 'benchmark_irr_comparison' in extracted_data:
            percentiles.append(extracted_data['benchmark_irr_comparison']['percentile'])
        
        if 'benchmark_moic_comparison' in extracted_data:
            percentiles.append(extracted_data['benchmark_moic_comparison']['percentile'])
        
        if 'benchmark_dpi_comparison' in extracted_data:
            percentiles.append(extracted_data['benchmark_dpi_comparison']['percentile'])
        
        # If benchmark comparisons are available, use them
        if percentiles:
            top_performances = ['top_decile', 'top_quartile']
            bottom_performances = ['bottom_decile', 'bottom_quartile']
            
            top_count = sum(1 for p in percentiles if p in top_performances)
            bottom_count = sum(1 for p in percentiles if p in bottom_performances)
            
            if top_count > bottom_count and top_count >= len(percentiles) / 2:
                return 'Outperforming'
            elif bottom_count > top_count and bottom_count >= len(percentiles) / 2:
                return 'Underperforming'
            else:
                return 'As Expected'
        
        # If benchmarks unavailable (no vintage or asset_class), use combined quantitative + qualitative assessment
        return self._assess_performance_without_benchmarks(extracted_data, original_performance_summary)
    
    def _assess_performance_without_benchmarks(self, extracted_data, original_performance_summary=None):
        """
        Assess performance using both quantitative metrics and qualitative information from the document.
        
        Args:
            extracted_data: The extracted data dictionary
            original_performance_summary: The performance summary extracted from the document
            
        Returns:
            Performance summary: 'Outperforming', 'Underperforming', or 'As Expected'
        """
        # Quantitative assessment based on metrics
        net_irr = extracted_data.get('net_irr')
        net_moic = extracted_data.get('net_moic')
        net_dpi = extracted_data.get('net_dpi')
        
        quantitative_scores = []
        
        # Reasonable thresholds for private investments:
        # IRR: <5% = Underperforming, 5-12% = As Expected, >12% = Outperforming
        # MOIC: <1.2x = Underperforming, 1.2-2.0x = As Expected, >2.0x = Outperforming
        # DPI: <0.5x = Underperforming, 0.5-1.0x = As Expected, >1.0x = Outperforming
        
        if net_irr is not None:
            if net_irr < 5:
                quantitative_scores.append('underperforming')
            elif net_irr > 12:
                quantitative_scores.append('outperforming')
            else:
                quantitative_scores.append('as_expected')
        
        if net_moic is not None:
            if net_moic < 1.2:
                quantitative_scores.append('underperforming')
            elif net_moic > 2.0:
                quantitative_scores.append('outperforming')
            else:
                quantitative_scores.append('as_expected')
        
        if net_dpi is not None:
            if net_dpi < 0.5:
                quantitative_scores.append('underperforming')
            elif net_dpi > 1.0:
                quantitative_scores.append('outperforming')
            else:
                quantitative_scores.append('as_expected')
        
        # Qualitative assessment from document content
        qualitative_score = self._assess_qualitative_performance(extracted_data, original_performance_summary)
        
        # Combine quantitative and qualitative assessments
        all_scores = quantitative_scores.copy()
        if qualitative_score:
            all_scores.append(qualitative_score)
        
        # If we have any scores, determine based on majority
        if all_scores:
            underperforming_count = sum(1 for s in all_scores if s == 'underperforming')
            outperforming_count = sum(1 for s in all_scores if s == 'outperforming')
            
            if underperforming_count > outperforming_count:
                return 'Underperforming'
            elif outperforming_count > underperforming_count:
                return 'Outperforming'
            else:
                return 'As Expected'
        
        # Fallback to original performance summary from document if no metrics available
        if original_performance_summary and original_performance_summary in ['Outperforming', 'Underperforming', 'As Expected']:
            return original_performance_summary
        
        return 'As Expected'
    
    def _assess_qualitative_performance(self, extracted_data, original_performance_summary=None):
        """
        Assess performance qualitatively based on language and content in the document.
        Uses LLM to analyze the qualitative aspects when quantitative metrics alone aren't sufficient.
        
        Args:
            extracted_data: The extracted data dictionary
            original_performance_summary: The performance summary extracted from the document
            
        Returns:
            'underperforming', 'outperforming', 'as_expected', or None
        """
        # Check original performance summary first
        if original_performance_summary:
            original_lower = original_performance_summary.lower()
            if 'outperforming' in original_lower or 'strong' in original_lower or 'excellent' in original_lower:
                return 'outperforming'
            elif 'underperforming' in original_lower or 'weak' in original_lower or 'challenging' in original_lower or 'disappointing' in original_lower:
                return 'underperforming'
            elif 'as expected' in original_lower or 'meeting' in original_lower or 'on track' in original_lower:
                return 'as_expected'
        
        # Combine all text sections for qualitative analysis
        all_text = []
        
        for section in ['investment_performance', 'key_takeaways', 'business_updates']:
            items = extracted_data.get(section, [])
            if isinstance(items, list):
                all_text.extend(items)
            elif isinstance(items, str):
                all_text.append(items)
        
        if not all_text:
            return None
        
        combined_text = '\n'.join(all_text)
        
        # Use LLM to assess qualitative performance
        try:
            qualitative_prompt = """Based on the following investment update content, assess the overall performance sentiment. Consider:
- Language used (positive, negative, neutral)
- Management commentary tone
- Market conditions described
- Portfolio performance descriptions
- Strategic outlook

Return ONLY one word: "Outperforming", "Underperforming", or "AsExpected"

Content:
{content}"""

            response = self.client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": "You are an investment analyst assessing fund performance based on qualitative indicators in investor updates."},
                    {"role": "user", "content": qualitative_prompt.format(content=combined_text)}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if 'outperforming' in result:
                return 'outperforming'
            elif 'underperforming' in result:
                return 'underperforming'
            elif 'asexpected' in result or 'as expected' in result:
                return 'as_expected'
            
        except Exception as e:
            print(f"Error in qualitative assessment: {e}")
            # Fallback to keyword-based analysis
            return self._assess_qualitative_keywords(combined_text)
        
        return None
    
    def _assess_qualitative_keywords(self, text):
        """Fallback keyword-based qualitative assessment."""
        text_lower = text.lower()
        
        # Positive performance indicators
        positive_indicators = [
            'strong performance', 'outperforming', 'exceeded', 'above expectations',
            'record', 'best', 'top', 'excellent', 'outstanding', 'significant gains',
            'successful', 'bullish', 'improving', 'positive momentum', 'strong returns'
        ]
        
        # Negative performance indicators
        negative_indicators = [
            'underperforming', 'below expectations', 'challenging', 'disappointing',
            'weak', 'declining', 'concerns', 'headwinds', 'difficult', 'struggling',
            'losses', 'negative', 'bearish', 'deteriorating', 'under pressure'
        ]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        # If clear qualitative signals, use them
        if positive_count > negative_count and positive_count >= 2:
            return 'outperforming'
        elif negative_count > positive_count and negative_count >= 2:
            return 'underperforming'
        elif positive_count == negative_count and (positive_count > 0 or negative_count > 0):
            return 'as_expected'
        
        return None


class FormattingAgent:
    """Agent responsible for formatting extracted information into the standardized update format."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    def format_update(self, extracted_data):
        """
        Format extracted information (JSON) into the standardized update format.
        Ensures each section does not exceed 200 words.
        
        Args:
            extracted_data: The extracted JSON data dict from ExtractionAgent
            
        Returns:
            Tuple of (formatted_update_text, metrics_dict)
        """
        start_time = time.time()
        
        try:
            # Convert JSON to string for the LLM
            json_str = json.dumps(extracted_data, indent=2)
            
            # Add note about benchmark comparisons if they exist
            benchmark_note = ""
            if any(key in extracted_data for key in ['benchmark_irr_comparison', 'benchmark_moic_comparison', 'benchmark_dpi_comparison']):
                benchmark_note = "\n\nNote: Benchmark comparison data is included in the JSON. Format IRR/MOIC/DPI lines with inline benchmark comparisons like: 'Net IRR: 15.5% (vs benchmark 13.1% - Above Median)'"
            
            user_message = f"Format the following extracted investment information (JSON) into the standardized format:\n\n{json_str}\n\nRemember: Each section must not exceed 200 words. Prioritize performance metrics and new developments over generic overview information.{benchmark_note}"
            
            response = self.client.chat.completions.create(
                model=FORMATTING_MODEL,
                messages=[
                    {"role": "system", "content": FORMATTING_PROMPT},
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Track token usage
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                total_tokens = response.usage.total_tokens or 0
            
            latency_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Calculate cost for formatting agent (gpt-5-mini)
            # Pricing: Input $0.25/1M tokens, Output $2.00/1M tokens
            formatting_cost = (prompt_tokens / 1_000_000 * 0.25) + (completion_tokens / 1_000_000 * 2.00)
            
            metrics = {
                'formatting_tokens': total_tokens,
                'formatting_prompt_tokens': prompt_tokens,
                'formatting_completion_tokens': completion_tokens,
                'formatting_cost': formatting_cost,
                'formatting_latency_ms': latency_ms
            }
            
            formatted_text = response.choices[0].message.content.strip()
            
            # Verify and enforce 200-word limit per section
            formatted_text = self._enforce_word_limits(formatted_text)
            
            return formatted_text, metrics
        except Exception as e:
            print(f"Error formatting update with OpenAI: {e}")
            raise
    
    def _enforce_word_limits(self, text):
        """
        Ensure each section does not exceed 200 words.
        If a section exceeds the limit, truncate it intelligently.
        """
        lines = text.split('\n')
        result_lines = []
        current_section = []
        current_section_name = None
        in_bullet_section = False
        
        for line in lines:
            # Check if this is a section header
            is_section_header = (
                line.strip().endswith(':') and 
                not line.startswith('  ') and
                (line.startswith('Investment Performance:') or 
                 line.startswith('Key Takeaways:') or 
                 line.startswith('Business Updates/Market Commentary:'))
            )
            
            if is_section_header:
                # Process previous section if exists
                if current_section:
                    result_lines.extend(self._limit_section_words(current_section, 200))
                
                # Start new section
                result_lines.append(line)
                current_section = []
                current_section_name = line.strip()
                in_bullet_section = True
            elif in_bullet_section:
                if line.startswith('  • '):
                    current_section.append(line)
                elif line.strip() == '':
                    # End of bullet section
                    if current_section:
                        result_lines.extend(self._limit_section_words(current_section, 200))
                        result_lines.append('')
                    current_section = []
                    in_bullet_section = False
                else:
                    # Non-bullet line - add to result
                    result_lines.append(line)
            else:
                # Outside bullet sections - add directly
                result_lines.append(line)
        
        # Process final section if exists
        if current_section:
            result_lines.extend(self._limit_section_words(current_section, 200))
        
        return '\n'.join(result_lines)
    
    def _limit_section_words(self, bullet_lines, max_words):
        """
        Limit a section's bullet points to not exceed max_words.
        Prioritizes earlier bullets (assumed to be more important).
        """
        total_words = sum(len(line.split()) for line in bullet_lines)
        
        if total_words <= max_words:
            return bullet_lines
        
        # Count words per bullet and prioritize
        bullet_word_counts = [(i, len(line.split())) for i, line in enumerate(bullet_lines)]
        bullet_word_counts.sort(key=lambda x: x[0])  # Keep original order
        
        result_lines = []
        current_words = 0
        
        for i, word_count in bullet_word_counts:
            if current_words + word_count <= max_words:
                result_lines.append(bullet_lines[i])
                current_words += word_count
            else:
                # Try to fit partial bullet if possible
                remaining_words = max_words - current_words
                if remaining_words > 10:  # Only truncate if meaningful space remains
                    words = bullet_lines[i].split()
                    truncated = ' '.join(words[:remaining_words]) + '...'
                    result_lines.append(truncated)
                break
        
        return result_lines


class AnalysisAgent:
    """Orchestrates the extraction and formatting agents to analyze investment updates."""
    
    def __init__(self):
        self.extraction_agent = ExtractionAgent()
        self.formatting_agent = FormattingAgent()
    
    def analyze_update(self, text, investment_name=None):
        """
        Analyze an investment update and return formatted text, metadata, and metrics.
        
        Args:
            text: The text content from the PDF
            investment_name: Optional name of the investment
            
        Returns:
            Tuple of (formatted_update_text, metadata_dict, metrics_dict)
        """
        # Step 1: Extract information
        extracted_data, metadata, extraction_metrics = self.extraction_agent.extract_information(text, investment_name)
        
        # Step 2: Format the extracted information
        formatted_text, formatting_metrics = self.formatting_agent.format_update(extracted_data)
        
        # Combine all metrics
        all_metrics = {**extraction_metrics, **formatting_metrics}
        
        # Calculate total cost
        all_metrics['total_cost'] = extraction_metrics.get('extraction_cost', 0) + formatting_metrics.get('formatting_cost', 0)
        
        return formatted_text, metadata, all_metrics
