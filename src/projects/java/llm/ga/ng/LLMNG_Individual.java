package llm.ga.ng;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.TreeMap;

import ga.GAIndividual;
import llm.LLMNG;
import llm.LLM_Lucas_CV.function;
import llm.WeightedErrorSorter;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;

public class LLMNG_Individual implements GAIndividual<LLMNG_Individual> {
		
	private static int nrNeurons = 12;
	
	final static Map<String,Object[]> params;
		
	static {
		params = new TreeMap<>();
		
		List<Object> list = new ArrayList<>();
		for(int l : new int[]{ 100000 } )
			list.add( l );
		params.put("t_max", list.toArray(new Object[]{}));
				
		list = new ArrayList<>();
		/*for(double l = 0; l <= 1.01; l+=0.05 )
			list.add( l );*/
		/*for(int l = 1; l <= nrNeurons; l++ )
			list.add( l );*/
		//list.add(nrNeurons);
		list.add( 0.0 );
		params.put("w", list.toArray(new Object[]{}));
				
		// quantization parameters:
		
		list = new ArrayList<>();
		for( double nb1Init : new double[] { 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 6, 9 } )
			list.add(nb1Init);
		params.put("nb1Init", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double nb1Final : new double[] { 0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1 } )
			list.add(nb1Final);
		params.put("nb1Final", list.toArray(new Object[]{}));

		list = new ArrayList<>();
		for( function nb1Func : new function[] { function.Power, function.Linear } )
			list.add(nb1Func);
		params.put("nb1Func", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr1Init : new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 } )
			list.add(lr1Init);
		params.put("lr1Init", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr1Final : new double[] { 0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1 } )
			list.add(lr1Final);
		params.put("lr1Final", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( function lr1Func : new function[] { function.Power, function.Linear } )
			list.add(lr1Func);
		params.put("lr1Func", list.toArray(new Object[]{}));
				
		// coefficient parameters:
		
		list = new ArrayList<>();
		for( boolean b : new boolean[] { true, false } )
			list.add(b);
		params.put("aMode", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( boolean b : new boolean[] { true, false } )
			list.add(b);
		params.put("uMode", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double nb2Init : new double[] { 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 6, 9 } )
			list.add(nb2Init);
		params.put("nb2Init", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double nb2Final : new double[] { 0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1 } )
			list.add(nb2Final);
		params.put("nb2Final", list.toArray(new Object[]{}));

		list = new ArrayList<>();
		for( function nb2Func : new function[] { function.Power, function.Linear } )
			list.add(nb2Func);
		params.put("nb2Func", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr2Init : new double[] { 0.05, 0.1, 0.2, 0.3, 0.4, 0.5 } )
			list.add(lr2Init);
		params.put("lr2Init", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr2Final : new double[] { 0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1 } )
			list.add(lr2Final);
		params.put("lr2Final", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( function lr2Func : new function[] { function.Power, function.Linear } )
			list.add(lr2Func);
		params.put("lr2Func", list.toArray(new Object[]{}));
	}
	
	public Map<String,Object> iParam;
	private Random r = new Random();
	
	public LLMNG_Individual() {	
		iParam = new TreeMap<>();
		for( Entry<String, Object[]> e : params.entrySet() ) {
			Object[] o = e.getValue();
			iParam.put(e.getKey(), o[r.nextInt(o.length)] );
		}
	}

	public LLMNG_Individual(Map<String, Object> nParam) {
		iParam = new TreeMap<>(nParam);
	}
	
	public LLMNG_Individual(String s ) {
		iParam = new TreeMap<>();
		s = s.replace("{", "");
		s = s.replaceAll("}",  "");
		
		StringTokenizer st = new StringTokenizer(s, ",");
		while( st.hasMoreTokens() ) {
			String nt = st.nextToken();
			String[] sp = nt.split("=");
			sp[0] = sp[0].replace(" ", "");
			try {
				iParam.put(sp[0], Integer.parseInt(sp[1]) ); 	
				continue;
			} catch (NumberFormatException e) {	}
			try {
				iParam.put(sp[0], Double.parseDouble(sp[1]) ); 	
				continue;
			} catch (NumberFormatException e) { }
			try {
				iParam.put(sp[0], function.valueOf(sp[1]) );
				continue;
			} catch (IllegalArgumentException e) { }
			try {
				iParam.put(sp[0], Boolean.valueOf(sp[1]) );
				continue;
			} catch (IllegalArgumentException e) { }
			try {
				iParam.put(sp[0], String.valueOf(sp[1]) );
				continue;
			} catch (IllegalArgumentException e) { }
		}
	}

	@Override
	public LLMNG_Individual mutate() {
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
		
		return new LLMNG_Individual(nParam);
	}

	@Override
	public LLMNG_Individual recombine(LLMNG_Individual mother) {
		Map<String,Object> nParam = new HashMap<>();
		for( String key : mother.iParam.keySet() )
			if( r.nextBoolean() )
				nParam.put(key, mother.iParam.get(key) );
			else
				nParam.put(key, iParam.get(key) );
		
		return new LLMNG_Individual(nParam);
	}
	
	public String toString() {
		return iParam.toString();
	}
	
	public LLMNG train(List<double[]> samples, int[] fa, int[] ga, int ta ) {
		return train(samples,fa,ga,ta,0);
	}
		
	public LLMNG train(List<double[]> samples, int[] fa, int[] ga, int ta, int seed ) {
		Random r1 = new Random(0);
		
		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(ga);
		WeightedErrorSorter wes =  new WeightedErrorSorter(null, fDist, samples, ta, (double)iParam.get("w"));
		//Sorter<double[]> wes = new DefaultSorter<>(fDist);
		//Sorter<double[]> wes = new KangasSorter<double[]>(gDist, fDist, (int)iParam.get("w"));
		
		List<double[]> neurons = new ArrayList<>();
		while (neurons.size() < LLMNG_Individual.nrNeurons ) {
			int idx = r1.nextInt(samples.size());
			double[] d = samples.get(idx);
			neurons.add( Arrays.copyOf(d,d.length) );
		}
		
		LLMNG llmng = new LLMNG(neurons, 
				getFunction( (double)iParam.get("nb1Init"), (double)iParam.get("nb1Final"), (function)iParam.get("nb1Func")),
				getFunction( (double)iParam.get("lr1Init"), (double)iParam.get("lr1Final"), (function)iParam.get("lr1Func")),
				getFunction( (double)iParam.get("nb2Init"), (double)iParam.get("nb2Final"), (function)iParam.get("nb2Func")),
				getFunction( (double)iParam.get("lr2Init"), (double)iParam.get("lr2Final"), (function)iParam.get("lr2Func")),
				wes, fa, 1);
		llmng.aMode = (boolean)iParam.get("aMode");
		llmng.uMode = (boolean)iParam.get("uMode");
		llmng.ignSupport = false;
		wes.setSupervisedNet(llmng);
		
		int t_max = (int)iParam.get("t_max");
		for (int t = 0; t < t_max; t++) {
			int j = r1.nextInt( samples.size() );
			double[] d = samples.get(j);
			llmng.train((double) t / t_max, d, new double[] { d[ta] } );
		}
		return llmng;
	}
	
	private DecayFunction getFunction(double init, double fin, function func) {
		if (func == function.Power)
			return new PowerDecay(init, fin);
		else if (func == function.Linear)
			return new LinearDecay(init, fin);
		else
			throw new RuntimeException("Unkown function");
	}
}
