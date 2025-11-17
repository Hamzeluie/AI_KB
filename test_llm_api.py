import asyncio
import aiohttp
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

@dataclass
class RequestResult:
    user_id: int
    request_id: int
    start_time: float
    end_time: float
    response_time: float
    tokens_generated: int
    status: str

class VLLMPerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8210"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def send_request(self, user_id: int, request_id: int, messages: List[Dict], 
                          max_tokens: int = 512, temperature: float = 0.7) -> RequestResult:
        start_time = time.time()
        
        try:
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            async with self.session.post(f"{self.base_url}/v1/chat/completions", 
                                       json=payload) as response:
                data = await response.json()
                end_time = time.time()
                
                # Extract tokens generated (approximate)
                content = data['choices'][0]['message']['content']
                tokens_generated = len(content.split())  # Rough estimation
                
                return RequestResult(
                    user_id=user_id,
                    request_id=request_id,
                    start_time=start_time,
                    end_time=end_time,
                    response_time=end_time - start_time,
                    tokens_generated=tokens_generated,
                    status="success" if response.status == 200 else "error"
                )
        except Exception as e:
            end_time = time.time()
            return RequestResult(
                user_id=user_id,
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                response_time=end_time - start_time,
                tokens_generated=0,
                status=f"error: {str(e)}"
            )

    async def run_concurrent_test(self, num_users: int, requests_per_user: int, 
                                messages: List[Dict], config: Dict[str, Any]) -> List[RequestResult]:
        """Run concurrent test with multiple users."""
        tasks = []
        
        for user_id in range(num_users):
            for req_id in range(requests_per_user):
                task = self.send_request(
                    user_id=user_id,
                    request_id=req_id,
                    messages=messages,
                    max_tokens=config.get('max_tokens', 512),
                    temperature=config.get('temperature', 0.7)
                )
                tasks.append(task)
                # Small delay to simulate realistic usage
                await asyncio.sleep(0.01)
        
        results = await asyncio.gather(*tasks)
        return results

def generate_test_messages(prompt: str = "Explain quantum computing in simple terms.") -> List[Dict]:
    return [
        {"role": "user", "content": prompt}
    ]

def analyze_results(results: List[RequestResult]) -> Dict[str, Any]:
    """Analyze performance metrics from test results."""
    successful_results = [r for r in results if r.status == "success"]
    
    if not successful_results:
        return {
            "total_requests": len(results),
            "successful_requests": 0,
            "failed_requests": len(results),
            "avg_response_time": 0,
            "min_response_time": 0,
            "max_response_time": 0,
            "p95_response_time": 0,
            "throughput_requests_per_second": 0,
            "avg_tokens_per_second": 0
        }
    
    response_times = [r.response_time for r in successful_results]
    total_time = max(r.end_time for r in successful_results) - min(r.start_time for r in successful_results)
    
    return {
        "total_requests": len(results),
        "successful_requests": len(successful_results),
        "failed_requests": len(results) - len(successful_results),
        "avg_response_time": np.mean(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "p95_response_time": np.percentile(response_times, 95),
        "throughput_requests_per_second": len(successful_results) / total_time if total_time > 0 else 0,
        "avg_tokens_per_second": np.mean([r.tokens_generated / r.response_time for r in successful_results if r.response_time > 0]) if successful_results else 0
    }

async def run_performance_tests():
    """Run performance tests with different VLLM configurations."""
    test_configs = [
        {"name": "Low Load", "num_users": 2, "requests_per_user": 3, "max_tokens": 256, "temperature": 0.7},
        {"name": "Medium Load", "num_users": 5, "requests_per_user": 4, "max_tokens": 512, "temperature": 0.7},
        {"name": "High Load", "num_users": 10, "requests_per_user": 5, "max_tokens": 1024, "temperature": 0.7},
        {"name": "High Temp", "num_users": 5, "requests_per_user": 4, "max_tokens": 512, "temperature": 1.0},
        {"name": "Low Temp", "num_users": 5, "requests_per_user": 4, "max_tokens": 512, "temperature": 0.1}
    ]
    
    messages = generate_test_messages()
    results_summary = []
    
    async with VLLMPerformanceTester() as tester:
        for config in test_configs:
            print(f"\nRunning test: {config['name']}")
            print(f"Users: {config['num_users']}, Requests per user: {config['requests_per_user']}")
            print(f"Max tokens: {config['max_tokens']}, Temperature: {config['temperature']}")
            
            start_time = time.time()
            results = await tester.run_concurrent_test(
                num_users=config['num_users'],
                requests_per_user=config['requests_per_user'],
                messages=messages,
                config=config
            )
            end_time = time.time()
            
            analysis = analyze_results(results)
            analysis['config_name'] = config['name']
            analysis['test_duration'] = end_time - start_time
            results_summary.append(analysis)
            
            print(f"Test completed in {analysis['test_duration']:.2f}s")
            print(f"Successful requests: {analysis['successful_requests']}/{analysis['total_requests']}")
            print(f"Average response time: {analysis['avg_response_time']:.2f}s")
            print(f"Throughput: {analysis['throughput_requests_per_second']:.2f} req/s")
            print("-" * 50)
    
    return results_summary

def plot_results(results_summary: List[Dict]):
    """Plot performance comparison across different configurations."""
    config_names = [r['config_name'] for r in results_summary]
    avg_response_times = [r['avg_response_time'] for r in results_summary]
    throughputs = [r['throughput_requests_per_second'] for r in results_summary]
    success_rates = [r['successful_requests'] / r['total_requests'] * 100 for r in results_summary]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Response time comparison
    axes[0, 0].bar(config_names, avg_response_times, color='skyblue')
    axes[0, 0].set_title('Average Response Time by Configuration')
    axes[0, 0].set_ylabel('Response Time (s)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Throughput comparison
    axes[0, 1].bar(config_names, throughputs, color='lightgreen')
    axes[0, 1].set_title('Throughput by Configuration')
    axes[0, 1].set_ylabel('Requests/Second')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Success rate comparison
    axes[1, 0].bar(config_names, success_rates, color='orange')
    axes[1, 0].set_title('Success Rate by Configuration')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Response time vs Throughput scatter
    axes[1, 1].scatter(avg_response_times, throughputs, s=100, alpha=0.7)
    for i, config_name in enumerate(config_names):
        axes[1, 1].annotate(config_name, (avg_response_times[i], throughputs[i]))
    axes[1, 1].set_xlabel('Average Response Time (s)')
    axes[1, 1].set_ylabel('Throughput (req/s)')
    axes[1, 1].set_title('Response Time vs Throughput')
    
    plt.tight_layout()
    plt.savefig('vllm_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_analysis(results_summary: List[Dict]):
    """Print detailed analysis of results."""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    for result in results_summary:
        print(f"\nConfiguration: {result['config_name']}")
        print(f"  Total Requests: {result['total_requests']}")
        print(f"  Successful Requests: {result['successful_requests']}")
        print(f"  Failed Requests: {result['failed_requests']}")
        print(f"  Success Rate: {result['successful_requests']/result['total_requests']*100:.2f}%")
        print(f"  Average Response Time: {result['avg_response_time']:.3f}s")
        print(f"  Min Response Time: {result['min_response_time']:.3f}s")
        print(f"  Max Response Time: {result['max_response_time']:.3f}s")
        print(f"  P95 Response Time: {result['p95_response_time']:.3f}s")
        print(f"  Throughput: {result['throughput_requests_per_second']:.2f} req/s")
        print(f"  Avg Tokens/Second: {result['avg_tokens_per_second']:.2f} tok/s")
        print(f"  Test Duration: {result['test_duration']:.2f}s")

async def main():
    print("Starting VLLM Performance Test...")
    print("Make sure your VLLM server is running at http://localhost:8210")
    
    try:
        results = await run_performance_tests()
        print_detailed_analysis(results)
        plot_results(results)
        
        # Save results to JSON file
        with open('performance_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("\nResults saved to performance_results.json")
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(main())