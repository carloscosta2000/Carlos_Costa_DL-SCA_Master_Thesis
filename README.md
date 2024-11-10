# Master Thesis - Break and Protect Modern CPUs using Machine Learning @ INESC-ID
## by Carlos Costa IST Student Number - 105768 
### Advisors
  * Prof. Ricardo Chaves
  * Prof. Aleksandar Ilić

**Final Grade** 18/20

### Abstract
Cryptographic algorithms with sensitive information are of extended use in today’s digital landscape,
making their security and protection a crucial topic. One of the attacks that pose a threat to these algorithms are Side Channel Attacks, which exploit information unintentionally leaked from the algorithm’s
implementation, such as power consumption, to infer sensitive data. Recently, these attacks have been
studied with the help of Deep Learning algorithms, with results showing their prowess in breaking cryptographic implementations. A new avenue for obtaining power traces that does not require physical access
to the victim’s device has opened up with the introduction of Intel’s Running Average Power Limit interface, which has the purpose of monitoring and reporting the power consumption of a device, however,
the sampling rate of the power traces obtained is seriously hindered.
In this work, we present a framework that incorporates known state-of-the-art Deep Learning techniques in the low sampling rate context for the first time and evaluates the feasibility of this novel domain.
The results validate the interest in this domain, as certain configurations of the framework can accurately
predict the encryption key of the targeted algorithm, significantly reduce the power traces required for
this attack, and be effective in contexts where the attacker’s adversarial model is reduced. Additionally,
this work also discusses the methodology behind power acquisition for Side Channel Attacks in this
context, hypothesizing possible improvements to streamline this process.

##### Keywords
Side Channel Attacks, Deep Learning, Running Average Power Limit, Low Sampling Rate


### Instructions
1. After `conda` is installed, create a new environment using the command: `conda env create -f conda_environment.yml`
2. Then, activate the environment using `conda activate DLSCA`
3. Lastly, with correct power trace file system hierarchy: `python3 dlsca.py`
