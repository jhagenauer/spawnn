package spawnn.gui;

import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;

import net.miginfocom.swing.MigLayout;

public class DistanceDialog extends JDialog implements ActionListener {

	JComboBox distMode, statMode;
	
	JButton ok, cancel;
	boolean okPressed = false;

	public static enum StatMode {Mean, Median, Variance, Min, Max};
	public static enum DistMode {Normal, Geo };
	
	private static final long serialVersionUID = -6577606485481195079L;

	public DistanceDialog(Frame parent, String string, boolean b, boolean enableGeo) {
		super(parent, string, b);

		setLayout(new MigLayout(""));

		add(new JLabel("Statistic:"));
		statMode = new JComboBox();
		statMode.setModel(new DefaultComboBoxModel(StatMode.values()));
		
		statMode.addActionListener(this);
		add(statMode, "wrap");
		
		add(new JLabel("Distance:"));
		distMode = new JComboBox();
		distMode.setModel(new DefaultComboBoxModel(DistMode.values()));
			
		distMode.addActionListener(this);
		distMode.setEnabled(enableGeo);
		add(distMode, "wrap");
		
		ok = new JButton("OK");
		ok.addActionListener(this);
		cancel = new JButton("Cancel");
		cancel.addActionListener(this);
		add(ok, "split 2, push");
		add(cancel);

		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		pack();
		setAlwaysOnTop(true);
		setLocationRelativeTo(parent);
		setVisible(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == ok) {
			okPressed = true;
			dispose();
		} else if( e.getSource() == cancel ){
			okPressed = false;
			dispose();
		}
	}

	public boolean isOkPressed() {
		return okPressed;
	}

	public DistMode getDistMode() {
		return (DistMode)distMode.getSelectedItem();
	}

	public StatMode getStatMode() {
		return (StatMode)statMode.getSelectedItem();
	}
}
