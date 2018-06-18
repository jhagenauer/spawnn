package llm.ga;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ga.GAIndividual;

import java.util.Random;

import llm.LLMNG;
import llm.LLM_Lucas_CV.function;

public class LLM_Individual implements GAIndividual<LLM_Individual> {
	
	final static int t_max = 100000;
	final static int nrNeurons = 25;
	
	final static Map<String,Object[]> params;
		
	static {
		params = new HashMap<>();
				
		List<Object> list = new ArrayList<>();
		for(int l = 1; l <= nrNeurons; l++ )
			list.add( l );
		params.put("l", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( LLMNG.mode mode : new LLMNG.mode[] { LLMNG.mode.fritzke, LLMNG.mode.martinetz } )
			list.add(mode);
		params.put("mode", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( LLMNG.mode mode : new LLMNG.mode[] { LLMNG.mode.fritzke, LLMNG.mode.martinetz } )
			list.add(mode);
		params.put("mode", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double nb1Init : new double[] { nrNeurons, nrNeurons * 2.0 / 3, nrNeurons * 1.0/3 } )
			list.add(nb1Init);
		params.put("nb1Inint", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double nb1Final : new double[] { 0.0001, 0.001, 0.01, 0.1, 1 } )
			list.add(nb1Final);
		params.put("nb1Final", list.toArray(new Object[]{}));

		list = new ArrayList<>();
		for( function nb1Func : new function[] { function.Power, function.Linear } )
			list.add(nb1Func);
		params.put("nb1Func", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr1Init : new double[] { 0.2, 0.4, 0.6, 0.8, 1 } )
			list.add(lr1Init);
		params.put("lr1Init", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr1Final : new double[] { 0.0001, 0.001, 0.01, 0.1 } )
			list.add(lr1Final);
		params.put("lr1Final", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( function lr1Func : new function[] { function.Power, function.Linear } )
			list.add(lr1Func);
		params.put("lr1Func", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( function lr1Func : new function[] { function.Power, function.Linear } )
			list.add(lr1Func);
		params.put("lr1Func", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double nb2Init : new double[] { nrNeurons, nrNeurons * 2.0 / 3, nrNeurons * 1.0/3 } )
			list.add(nb2Init);
		params.put("nb2Inint", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double nb2Final : new double[] { 0.0001, 0.001, 0.01, 0.1, 1 } )
			list.add(nb2Final);
		params.put("nb2Final", list.toArray(new Object[]{}));

		list = new ArrayList<>();
		for( function nb2Func : new function[] { function.Power, function.Linear } )
			list.add(nb2Func);
		params.put("nb2Func", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr2Init : new double[] { 0.2, 0.4, 0.6, 0.8, 1 } )
			list.add(lr2Init);
		params.put("lr2Init", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr2Final : new double[] { 0.0001, 0.001, 0.01, 0.1 } )
			list.add(lr2Final);
		params.put("lr2Final", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( function lr2Func : new function[] { function.Power, function.Linear } )
			list.add(lr2Func);
		params.put("lr2Func", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( function lr2Func : new function[] { function.Power, function.Linear } )
			list.add(lr2Func);
		params.put("lr2Func", list.toArray(new Object[]{}));
	}
	
	public Map<String,Object> iParam;
	private Random r = new Random();
	
	public LLM_Individual() {
		iParam = new HashMap<>();
		for( Entry<String, Object[]> e : params.entrySet() ) {
			Object[] o = e.getValue();
			iParam.put(e.getKey(), o[r.nextInt(o.length)] );
		}
	}

	public LLM_Individual(Map<String, Object> nParam) {
		iParam = new HashMap<>(nParam);
	}

	@Override
	public LLM_Individual mutate() {
		int sum = 0;
		for( Object[] o : params.values() )
			sum += o.length;
		int sel = r.nextInt(sum);
		
		String toMut = null;
		int start = 0;
		for( Entry<String, Object[]> e : params.entrySet() ) {
			if( start <= sel & sel < start + e.getValue().length ) {
				toMut = e.getKey();
				break;
			}
			start += e.getValue().length;
		}
		
		Map<String,Object> nParam = new HashMap<>(iParam);
		Object[] v = params.get(toMut);
		nParam.put(toMut, v[r.nextInt(v.length)]);
		
		return new LLM_Individual(nParam);
	}

	@Override
	public LLM_Individual recombine(LLM_Individual mother) {
		Map<String,Object> nParam = new HashMap<>();
		for( String key : mother.iParam.keySet() )
			if( r.nextBoolean() )
				nParam.put(key, mother.iParam.get(key) );
			else
				nParam.put(key, iParam.get(key) );
		
		return new LLM_Individual(nParam);
	}
}
