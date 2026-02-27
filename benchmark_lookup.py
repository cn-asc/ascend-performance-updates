"""Utility functions for looking up benchmark data."""
import json
from pathlib import Path


class BenchmarkLookup:
    """Helper class for looking up benchmark data by asset class and vintage."""
    
    def __init__(self, benchmarks_file="benchmarks.json"):
        """Initialize with benchmarks data file."""
        self.benchmarks_file = Path(benchmarks_file)
        self.data = self._load_benchmarks()
    
    def _load_benchmarks(self):
        """Load benchmarks from JSON file."""
        if not self.benchmarks_file.exists():
            print(f"Warning: Benchmarks file not found: {self.benchmarks_file}")
            return {}
        
        with open(self.benchmarks_file, 'r') as f:
            return json.load(f)
    
    def normalize_asset_class(self, asset_class):
        """Normalize asset class name to match benchmark keys."""
        if not asset_class:
            return None
            
        asset_class_lower = asset_class.lower().strip()
        
        # Map common variations to standard keys
        mapping = {
            'private equity': 'private_equity',
            'pe': 'private_equity',
            'venture capital': 'venture_capital',
            'vc': 'venture_capital',
            'real estate': 'real_estate',
            're': 'real_estate',
            'private debt': 'private_debt',
            'credit': 'private_debt',
            'debt': 'private_debt'
        }
        
        return mapping.get(asset_class_lower, asset_class_lower.replace(' ', '_'))
    
    def get_irr_benchmarks(self, asset_class, vintage):
        """
        Get IRR benchmarks for a given asset class and vintage.
        
        Args:
            asset_class: Asset class name (e.g., "Private Equity", "Venture Capital")
            vintage: Vintage year as string or int (e.g., "2020" or 2020)
            
        Returns:
            dict with IRR benchmark metrics, or None if not found
        """
        if not self.data:
            return None
            
        asset_key = self.normalize_asset_class(asset_class)
        vintage_str = str(vintage)
        
        if asset_key not in self.data:
            return None
        
        return self.data[asset_key].get('irrs_by_vintage', {}).get(vintage_str)
    
    def get_multiples_benchmarks(self, asset_class, vintage):
        """
        Get multiples (TVPI/DPI) benchmarks for a given asset class and vintage.
        
        Args:
            asset_class: Asset class name (e.g., "Private Equity", "Venture Capital")
            vintage: Vintage year as string or int (e.g., "2020" or 2020)
            
        Returns:
            dict with multiples benchmark metrics, or None if not found
        """
        if not self.data:
            return None
            
        asset_key = self.normalize_asset_class(asset_class)
        vintage_str = str(vintage)
        
        if asset_key not in self.data:
            return None
        
        return self.data[asset_key].get('multiples_by_vintage', {}).get(vintage_str)
    
    def compare_irr(self, asset_class, vintage, actual_irr):
        """
        Compare actual IRR against benchmarks.
        
        Args:
            asset_class: Asset class name
            vintage: Vintage year
            actual_irr: Actual IRR value (as percentage, e.g., 15.5 for 15.5%)
            
        Returns:
            dict with comparison results including 'median', 'percentile', 'category', or None if not found
        """
        benchmarks = self.get_irr_benchmarks(asset_class, vintage)
        if not benchmarks:
            return None
        
        median = benchmarks.get('median')
        if median is None:
            return None
        
        # Determine percentile
        if actual_irr >= benchmarks.get('top_decile', float('-inf')):
            percentile = 'top_decile'
            category = 'Top 10%'
        elif actual_irr >= benchmarks.get('top_quartile', float('-inf')):
            percentile = 'top_quartile'
            category = 'Top 25%'
        elif actual_irr >= benchmarks.get('median', float('-inf')):
            percentile = 'above_median'
            category = 'Above Median'
        elif actual_irr >= benchmarks.get('bottom_quartile', float('-inf')):
            percentile = 'below_median'
            category = 'Below Median'
        elif actual_irr >= benchmarks.get('bottom_decile', float('-inf')):
            percentile = 'bottom_quartile'
            category = 'Bottom 25%'
        else:
            percentile = 'bottom_decile'
            category = 'Bottom 10%'
        
        return {
            'median': median,
            'percentile': percentile,
            'category': category
        }
    
    def compare_moic(self, asset_class, vintage, actual_tvpi):
        """
        Compare actual MOIC (TVPI) against benchmarks.
        
        Args:
            asset_class: Asset class name
            vintage: Vintage year
            actual_tvpi: Actual TVPI value (e.g., 2.5 for 2.5x)
            
        Returns:
            dict with comparison results including 'median', 'percentile', 'category', or None if not found
        """
        benchmarks = self.get_multiples_benchmarks(asset_class, vintage)
        if not benchmarks:
            return None
        
        median = benchmarks.get('tvpi_median')
        if median is None:
            return None
        
        # Determine percentile
        if actual_tvpi >= benchmarks.get('tvpi_top_decile', float('-inf')):
            percentile = 'top_decile'
            category = 'Top 10%'
        elif actual_tvpi >= benchmarks.get('tvpi_top_quartile', float('-inf')):
            percentile = 'top_quartile'
            category = 'Top 25%'
        elif actual_tvpi >= benchmarks.get('tvpi_median', float('-inf')):
            percentile = 'above_median'
            category = 'Above Median'
        elif actual_tvpi >= benchmarks.get('tvpi_bottom_quartile', float('-inf')):
            percentile = 'below_median'
            category = 'Below Median'
        elif actual_tvpi >= benchmarks.get('tvpi_bottom_decile', float('-inf')):
            percentile = 'bottom_quartile'
            category = 'Bottom 25%'
        else:
            percentile = 'bottom_decile'
            category = 'Bottom 10%'
        
        return {
            'median': median,
            'percentile': percentile,
            'category': category
        }
    
    def compare_dpi(self, asset_class, vintage, actual_dpi):
        """
        Compare actual DPI against benchmarks.
        
        Args:
            asset_class: Asset class name
            vintage: Vintage year
            actual_dpi: Actual DPI value (e.g., 1.2 for 1.2x)
            
        Returns:
            dict with comparison results including 'median', 'percentile', 'category', or None if not found
        """
        benchmarks = self.get_multiples_benchmarks(asset_class, vintage)
        if not benchmarks:
            return None
        
        median = benchmarks.get('dpi_median')
        if median is None:
            return None
        
        # Determine percentile
        if actual_dpi >= benchmarks.get('dpi_top_decile', float('-inf')):
            percentile = 'top_decile'
            category = 'Top 10%'
        elif actual_dpi >= benchmarks.get('dpi_top_quartile', float('-inf')):
            percentile = 'top_quartile'
            category = 'Top 25%'
        elif actual_dpi >= benchmarks.get('dpi_median', float('-inf')):
            percentile = 'above_median'
            category = 'Above Median'
        elif actual_dpi >= benchmarks.get('dpi_bottom_quartile', float('-inf')):
            percentile = 'below_median'
            category = 'Below Median'
        elif actual_dpi >= benchmarks.get('dpi_bottom_decile', float('-inf')):
            percentile = 'bottom_quartile'
            category = 'Bottom 25%'
        else:
            percentile = 'bottom_decile'
            category = 'Bottom 10%'
        
        return {
            'median': median,
            'percentile': percentile,
            'category': category
        }


# Example usage and validation
if __name__ == "__main__":
    import sys
    
    lookup = BenchmarkLookup()
    
    if not lookup.data:
        print("❌ Error: benchmarks.json not found or empty")
        sys.exit(1)
    
    print("✅ benchmarks.json loaded successfully\n")
    
    # Validate structure
    asset_classes = ['private_equity', 'venture_capital', 'real_estate', 'private_debt']
    print("📊 Benchmark Data Status:")
    print("-" * 60)
    
    for asset_class in asset_classes:
        if asset_class not in lookup.data:
            print(f"❌ {asset_class}: Missing from benchmarks.json")
            continue
        
        irrs = lookup.data[asset_class].get('irrs_by_vintage', {})
        multiples = lookup.data[asset_class].get('multiples_by_vintage', {})
        
        irr_status = f"✅ {len(irrs)} vintages" if irrs else "⚠️  EMPTY"
        mult_status = f"✅ {len(multiples)} vintages" if multiples else "⚠️  EMPTY"
        
        print(f"{asset_class.replace('_', ' ').title():20} IRRs: {irr_status:20} Multiples: {mult_status}")
    
    print("\n" + "-" * 60)
    
    # Example lookups
    print("\n🔍 Example Lookups:")
    print("-" * 60)
    
    # Test PE IRR lookup
    pe_irr = lookup.get_irr_benchmarks("Private Equity", 2020)
    if pe_irr:
        print(f"Private Equity 2020 IRR Median: {pe_irr.get('median')}%")
        comparison = lookup.compare_irr("Private Equity", 2020, 15.5)
        if comparison:
            print(f"  → 15.5% IRR is {comparison['category']} (median: {comparison['median']}%)")
    
    # Test VC Multiples lookup
    vc_mult = lookup.get_multiples_benchmarks("Venture Capital", 2020)
    if vc_mult:
        print(f"\nVenture Capital 2020 TVPI Median: {vc_mult.get('tvpi_median')}x")
        comparison = lookup.compare_moic("Venture Capital", 2020, 2.5)
        if comparison:
            print(f"  → 2.5x MOIC is {comparison['category']} (median: {comparison['median']}x)")
    
    print("\n" + "-" * 60)
    print("💡 Tip: Use BenchmarkLookup in your code like this:")
    print("   from benchmark_lookup import BenchmarkLookup")
    print("   lookup = BenchmarkLookup()")
    print("   result = lookup.compare_irr('Private Equity', 2020, 15.5)")
