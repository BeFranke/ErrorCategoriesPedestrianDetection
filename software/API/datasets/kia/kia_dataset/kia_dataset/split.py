"""doc
# kia_dataset.split

> The official dataset split.

## Installation/Setup

Follow the instructions in the main readme.

## Constants

There are three constants which each contain a List[str] of the sequence names which are allowed for use in the respective split.

The official splits per tranche per company:
* `TRAIN_{company}_TRANCHE_{num}`: Can be used to train neural networks.
* `VAL_{company}_TRANCHE_{num}`: Can be used to validate the neural network during training.
* `TEST_{company}_TRANCHE_{num}`: This split must not be used during development and can only be used (ideally) one time once the development is done.

The official splits for the releases:
* `TRAIN_RELEASE_{num}`: Can be used to train neural networks.
* `VAL_RELEASE_{num}`: Can be used to validate the neural network during training.
* `TEST_RELEASE_{num}`: This split must not be used during development and can only be used (ideally) one time once the development is done.
"""

# Official splits per tranche of dataproducer
TRAIN_BIT_TRANCHE_2 = ["bit_results_sequence_0025-4beaace23b6d4d05a0373f2b0973f1f1", "bit_results_sequence_0026-756f83ad3bf04f74910af85f3e1edbfc", "bit_results_sequence_0027-b82518c4082a4180ae1a155cde5714ce", "bit_results_sequence_0028-d2a92fc6c8854c7a843d8218b70d3850", "bit_results_sequence_0029-4866ce99e9074b58a215408ca46c5d3a", "bit_results_sequence_0030-fc34c3fd51f448058516390af529ced4", "bit_results_sequence_0031-ef950d0778e44e9a8e5ad0b51c924b56", "bit_results_sequence_0032-e55bfe3e76e74c239657de86bcddb3d9", "bit_results_sequence_0033-1813b659171342f5a58729c5c326f0c3", "bit_results_sequence_0034-ce8e16f9af7049828b2e7018bd1edae7", "bit_results_sequence_0035-4c9cebe455fb484cb704c076a139009e", "bit_results_sequence_0036-3ddda3663f2744deb2730a3ff533a83a", "bit_results_sequence_0037-1a426b274409481ab9be6a9edcce5165", "bit_results_sequence_0038-8071ac915cc84150a822dd051307036c", "bit_results_sequence_0039-47a96215515f4ff7961fe3ceda93413c", "bit_results_sequence_0040-dfd2d3c14dc444e2a57b93221ece8f3b",
         "bit_results_sequence_0041-52f020a505ef4532aa35c7920c31192a", "bit_results_sequence_0042-ef2801d2f53948a68f66dd7abe87f08f", "bit_results_sequence_0043-914fbabecd554990aa43b7e25e76d364", "bit_results_sequence_0044-f275007ea89a4450afe2b74f23e29abc", "bit_results_sequence_0045-0a3a3672eabf43b2aa7c534e0c1434cd", "bit_results_sequence_0046-77c24d90fa1a479b9ddea4af9b6aa122", "bit_results_sequence_0047-2c228631b4854ab69a3bcdd8f2dc96a8", "bit_results_sequence_0048-c73af228b39c4a04985f3fab0a528f63", "bit_results_sequence_0049-553ac5ba347148f5a73acd16cd90cc97", "bit_results_sequence_0050-b4bf54803c2a4e898f988beff3c2031d", "bit_results_sequence_0051-956986134ccf4aecb2e11907db63dbc3", "bit_results_sequence_0052-996c54f40f7e4fbf8ff04a92ac8de874", "bit_results_sequence_0053-2dcb5cdd72754b4ea6861acaf855af24", "bit_results_sequence_0054-92c82ca99537483cba71a14ab3005802", "bit_results_sequence_0055-24f59334a9c04cef8e081e5a35f82bd5", "bit_results_sequence_0056-45645575f05d481696b41604166d0d40", "bit_results_sequence_0057-fcaa55bd97f5466a86bf1dfccd465f3b"]
VAL_BIT_TRANCHE_2 = ["bit_results_sequence_0058-9088659e1c2b4d4d9a560f1ad7bbcade", "bit_results_sequence_0059-66412eb6563244a4a8404d598989dafe", "bit_results_sequence_0060-07eecb79d532468d92db523c53219963",
       "bit_results_sequence_0061-46608f35d1554f70aae7839b1b395bd4", "bit_results_sequence_0062-2c07143eb7654524b9ccf600d06c1409", "bit_results_sequence_0063-045e48798e78468aad1e202f7369b0dc"]
TEST_BIT_TRANCHE_2 = ["bit_results_sequence_0064-62cc7801608e432fabe7d5dd31ef539a", "bit_results_sequence_0065-e89b45c30925426a8ff9c08b1aedd9df", "bit_results_sequence_0066-5cef52d687754a39927116d6cc2ce2f0",
        "bit_results_sequence_0067-2019f23b516a43ad905c7c26f79e3893", "bit_results_sequence_0068-97d348da5bcc432cbe596cc4ffbd3311", "bit_results_sequence_0069-9ee31866e95b47cca8f4f124379355a5"]

TRAIN_BIT_TRANCHE_3 = ["bit_results_sequence_0070-8de675d2f59e44a3aa68a6d450d9b949", "bit_results_sequence_0071-3e821f9967564c6ba1781cba477377a7", "bit_results_sequence_0072-7bd44c07de4840bf92a79ef9eeee66be", "bit_results_sequence_0073-2155f0496707449190bdb9e71531dccd", "bit_results_sequence_0074-7f7e186e218a483aa8c6de3c75684575", "bit_results_sequence_0075-5dd197410b6048c9a792113f279951a9", "bit_results_sequence_0076-f74f44e688d94d518a05e9ad3a9066d0", "bit_results_sequence_0079-ffc44024e73c4e189af9f64f4197b34b", "bit_results_sequence_0080-52c031bc9dc04c2d9810b52034bae35d", "bit_results_sequence_0081-9b65d36eb67f4eda80cb01323d0e66a6", "bit_results_sequence_0082-45838a1c393549e395bef509b72cac50", "bit_results_sequence_0083-5cc03ffadf74451696b2e76d20fbd3ae", "bit_results_sequence_0084-5de2fea8bf12432fa21466b4771e1de2", "bit_results_sequence_0085-13dd2ea45b3944558684ec03a25ac345", "bit_results_sequence_0086-67e9d73aad624f8b9066427fa2f7df82", "bit_results_sequence_0087-c5eb4a1e2c4f4468a1926edd695ebf10", "bit_results_sequence_0088-eadaa13c6efa4f20943bcf8faf8ab8b2",
                   "bit_results_sequence_0089-dd469e143995456faea6bf635b5841bf", "bit_results_sequence_0090-334fc4453a6144f8a6ff9956ecb9e864", "bit_results_sequence_0091-ed2d75b6bfb94b888e1ac2c10453286f", "bit_results_sequence_0092-ca68eb295c8e4e9e8c90615a6308d02e", "bit_results_sequence_0093-a3d4b834759141c39e15c5669032101b", "bit_results_sequence_0094-1858396943784168815b52bc18a8e6e6", "bit_results_sequence_0095-925287c8f7c0425f969740afae82a74a", "bit_results_sequence_0096-630ae7d8f2104df688a724f54db9db64", "bit_results_sequence_0097-91c01e5a5cb44b468138595a3229b7af", "bit_results_sequence_0098-563e8121394c42f4af05ebc4194792fe", "bit_results_sequence_0099-d1b7ae916f9746f7a778a25a57fb66db", "bit_results_sequence_0100-9e8811d4e8f245c384225fef0a10d8f0", "bit_results_sequence_0101-32581485420d44cda0dc9bc7cdbf2c54", "bit_results_sequence_0102-7479601eb04a4e3292b9b7a84a19ff46", "bit_results_sequence_0103-faa76a00b3d348af93c3f10a5de1b041", "bit_results_sequence_0105-e04f253a90ec42209b8a9e1a841b29c7", "bit_results_sequence_0107-6ed30e3c58044af1a79f3ee3f5c77271", "bit_results_sequence_0124-614c864d75674f33a7ccd036237b40fa"]
VAL_BIT_TRANCHE_3 = ["bit_results_sequence_0077-fd3a1bea3fc845d5bed8eb2b25df852c", "bit_results_sequence_0078-55a9e3d1ea32466cb23f1772e62c7051", "bit_results_sequence_0104-856339aaf1744da4b72621b2f5c220ea", "bit_results_sequence_0106-87a35332aaa74f4ba961941040a9fc22",
                  "bit_results_sequence_0108-7bc993c7935c4551b5c0629db336d28d", "bit_results_sequence_0114-b81ff102d0b343509f0378c84300c8f1", "bit_results_sequence_0116-33b646bb827e4bb2b65c7c52f482cc18", "bit_results_sequence_0125-513f98eacd314edeb49da18a2d6c74ec"]
TEST_BIT_TRANCHE_3 = ["bit_results_sequence_0109-564e8af4f42b458caa3aa989dc0e7da1", "bit_results_sequence_0110-b5a81bec38f445f0b5f5fb75bcb76d6c", "bit_results_sequence_0111-724ea5c61e6f4254a87e339625db44a6", "bit_results_sequence_0112-6422955af3ee49a289d2611b0e001489", "bit_results_sequence_0115-4876ec19e85c47bb8d19c6188237e78e", "bit_results_sequence_0117-4d70e7bbdd67472e8daff9a510f0f160", "bit_results_sequence_0118-c8643bda8d334ff381a2e4e16318d463",
                  "bit_results_sequence_0119-29c8120e3a384d7482a286cc89460e0b", "bit_results_sequence_0120-91f354c93d144d2e859ed3cb9a8c69f7", "bit_results_sequence_0121-6c23009bed0f4da29e5cc2c31abef8b1", "bit_results_sequence_0122-b9deaea30d6b4467b835407388cb8fbf", "bit_results_sequence_0123-267a5dc6210a445f8fac2a5fa68b49f4", "bit_results_sequence_0126-a27ead8dcc9443f699b634e9fe3a15e3", "bit_results_sequence_0127-b1b23f12104a45789b6bc19458579b82"]


# Official train/val/test sets for the releases.
TRAIN_RELEASE_1 = TRAIN_BIT_TRANCHE_2
VAL_RELEASE_1 = VAL_BIT_TRANCHE_2
TEST_RELEASE_1 = TEST_BIT_TRANCHE_2

TRAIN_RELEASE_2 = TRAIN_BIT_TRANCHE_2 + TRAIN_BIT_TRANCHE_3
VAL_RELEASE_2 = VAL_BIT_TRANCHE_2 + VAL_BIT_TRANCHE_3
TEST_RELEASE_2 = TEST_BIT_TRANCHE_2 + TEST_BIT_TRANCHE_3
