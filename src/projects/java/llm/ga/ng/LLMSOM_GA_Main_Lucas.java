package llm.ga.ng;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

import org.apache.log4j.Logger;

import heuristics.CostCalculator;
import heuristics.ga.GeneticAlgorithm;
import llm.LLMNG;
import nnet.SupervisedUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.SpatialDataFrame;

public class LLMSOM_GA_Main_Lucas {

	private static Logger log = Logger.getLogger(LLMSOM_GA_Main_Lucas.class);

	public static void main(String[] args) {

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/lucas/lucas.shp"), true);
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, sdf.size());

		int[] fa = new int[] { 
				2, // TLA
				3, // beds
				9, // rooms
				10, // lotsize
				19, // age
		};
		int[] ga = new int[] { 20, 21 };
		int ta = 0;

		for (double[] d : sdf.samples) {
			d[19] = Math.pow(d[19], 2);
			d[10] = Math.log(d[10]);
			d[2] = Math.log(d[2]);
			d[1] = Math.log(d[1]);
		}
		List<double[]> samples = sdf.samples;

		DataUtils.transform(samples, fa, Transform.zScore);
		//DataUtils.transform(samples, ta, Transform.zScore);

		CostCalculator<LLMNG_Individual> cc2 = new LLMNG_CV_CostCalculator(samples, fa, ga, ta);
		CostCalculator<LLMNG_Individual> cc2_qe = new LLMNG_QE_CostCalculator(samples, fa, ga, ta);

		for (LLMNG_Individual i : new LLMNG_Individual[] {
		}) {
			log.debug(i);
			log.debug(cc2.getCost(i) + " " + cc2_qe.getCost(i));

			LLMNG llmng = i.train(samples, fa, ga, ta);
			for (double[] n : llmng.getNeurons()) {
				// log.debug("n: "+Arrays.toString( strip(n,fa) )+", m:
				// "+Arrays.toString( strip( llmng.matrix.get(n)[0], fa) )+", o:
				// "+llmng.output.get(n)[0] );
			}
		}
		// System.exit(1);

		GeneticAlgorithm.tournamentSize = 3;
		GeneticAlgorithm.elitist = true;
		GeneticAlgorithm.recombProb = 0.7;

		List<LLMNG_Individual> init = new ArrayList<>();
		while (init.size() < 200) {
			init.add(new LLMNG_Individual());
		}

		GeneticAlgorithm<LLMNG_Individual> gen = new GeneticAlgorithm<LLMNG_Individual>();
		LLMNG_Individual result = (LLMNG_Individual) gen.search(init, cc2);

		log.info("best:");
		log.info(cc2.getCost(result));
		log.info(result.iParam);

	}

	public static double[] strip(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}
}
