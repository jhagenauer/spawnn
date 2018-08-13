package llm.ga.som;

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
import llm.LLMSOM;
import llm.LLM_Lucas_CV.function;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;

public class LLMSOM_Individual implements GAIndividual<LLMSOM_Individual> {
		
	final static Map<String,Object[]> params;
		
	static {
		params = new TreeMap<>();
		
		List<Object> list = new ArrayList<>();
		for(int l : new int[]{ 200000 } )
			list.add( l );
		params.put("t_max", list.toArray(new Object[]{}));
				
		list = new ArrayList<>();
		for(double l : new double[]{ 1, 2, 3, 4, 5, 6, 7, 8 } )
			list.add( l );
		params.put("w", list.toArray(new Object[]{}));
		
		// som parameters:
		
		list = new ArrayList<>();
		for( double nb1Init : new double[] { 8 } )
			list.add(nb1Init);
		params.put("nb1Init", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double nb1Final : new double[] { 0.1 } )
			list.add(nb1Final);
		params.put("nb1Final", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr1Init : new double[] { 0.8 } )
			list.add(lr1Init);
		params.put("lr1Init", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr1Final : new double[] { 0.001 } )
			list.add(lr1Final);
		params.put("lr1Final", list.toArray(new Object[]{}));
		
		// llm parameters:
		
		list = new ArrayList<>();
		for( boolean aMode : new boolean[] { true, false } )
			list.add(aMode);
		params.put("aMode", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( boolean uMode : new boolean[] { true, false } )
			list.add(uMode);
		params.put("uMode", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double nb2Init : new double[] { 1, 2, 3, 4, 5, 6, 7, 8 } )
			list.add(nb2Init);
		params.put("nb2Init", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double nb2Final : new double[] { 0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1 } )
			list.add(nb2Final);
		params.put("nb2Final", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr2Init : new double[] { 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5 } )
			list.add(lr2Init);
		params.put("lr2Init", list.toArray(new Object[]{}));
		
		list = new ArrayList<>();
		for( double lr2Final : new double[] { 0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1 } )
			list.add(lr2Final);
		params.put("lr2Final", list.toArray(new Object[]{}));
	}
	
	public Map<String,Object> iParam;
	private Random r = new Random();
	
	public LLMSOM_Individual() {	
		iParam = new TreeMap<>();
		for( Entry<String, Object[]> e : params.entrySet() ) {
			Object[] o = e.getValue();
			iParam.put(e.getKey(), o[r.nextInt(o.length)] );
		}
	}

	public LLMSOM_Individual(Map<String, Object> nParam) {
		iParam = new TreeMap<>(nParam);
	}
	
	public LLMSOM_Individual(String s ) {
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
				iParam.put(sp[0], Boolean.parseBoolean( sp[1]) );
				continue;
			} catch (IllegalArgumentException e) { }	
			/*try {
				iParam.put(sp[0], String.valueOf(sp[1]) );
				continue;
			} catch (IllegalArgumentException e) { }*/
		}
	}

	@Override
	public LLMSOM_Individual mutate() {
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
		
		return new LLMSOM_Individual(nParam);
	}

	@Override
	public LLMSOM_Individual recombine(LLMSOM_Individual mother) {
		Map<String,Object> nParam = new HashMap<>();
		for( String key : mother.iParam.keySet() )
			if( r.nextBoolean() )
				nParam.put(key, mother.iParam.get(key) );
			else
				nParam.put(key, iParam.get(key) );
		
		return new LLMSOM_Individual(nParam);
	}
	
	public String toString() {
		return iParam.toString();
	}
	
	public LLMSOM train(List<double[]> samples, int[] fa, int[] ga, int ta ) {
		return train(samples,fa,ga,ta,0);
	}
		
	public LLMSOM train(List<double[]> samples, int[] fa, int[] ga, int ta, int seed ) {
		Random r1 = new Random(seed);
		BmuGetter<double[]> bg = new KangasBmuGetter<>(new EuclideanDist(ga), new EuclideanDist(fa), (int)(double)iParam.get("w") );
		//BmuGetter<double[]> bg = new DefaultBmuGetter<>(new EuclideanDist(fa));

		Grid2D<double[]> grid = new Grid2DHex<>(18, 12);			
		for (GridPos p : grid.getPositions()) {
			double[] d = samples.get(r1.nextInt(samples.size()));
			grid.setPrototypeAt(p, Arrays.copyOf(d, d.length));
		}
		
		LLMSOM llmsom = new LLMSOM( 
				new GaussKernel( new LinearDecay( (double)iParam.get("nb1Init"), (double)iParam.get("nb1Final"))),
				new LinearDecay( (double)iParam.get("lr1Init"), (double)iParam.get("lr1Final")),
				grid,
				bg,
				new GaussKernel( new LinearDecay( (double)iParam.get("nb2Init"), (double)iParam.get("nb2Final"))),
				new LinearDecay( (double)iParam.get("lr2Init"), (double)iParam.get("lr2Final")),
				fa,1
				);
		llmsom.aMode = (boolean)iParam.get("aMode");
		llmsom.uMode = (boolean)iParam.get("uMode");
		
		for (int t = 0; t < (int)iParam.get("t_max"); t++) {
			int j = r1.nextInt( samples.size() );
			double[] d = samples.get(j);
			llmsom.train((double) t / (int)iParam.get("t_max"), d, new double[] { d[ta] } );
		}
		return llmsom;
	}
}
