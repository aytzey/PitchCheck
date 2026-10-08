# PitchServer kod, sağlamlık ve hız incelemesi — 4 Ekim 2026

`origin/main` üzerindeki en güncel commit (`cef084d`, 23 Eylül 2026) temel alındı. Değişiklikler `fix/server-runtime-hardening` branch'inde. Sunucudaki mevcut PitchServer dağıtımı güncellendi; diğer servislerin konteyner kimlikleri, başlangıç zamanları, restart sayıları ve sağlık durumları başlangıç kaydıyla karşılaştırıldı.

## Gerçek sunucu ve dağıtım

| Alan | Doğrulanan değer |
|---|---|
| Sunucu | `dkmserver@78.186.120.189`, SSH `2022` |
| Dizin | `/home/dkmserver/Desktop/Machinity/aytug/pitchserver` |
| GPU | NVIDIA RTX 5080; CUDA'nın gördüğü toplam bellek 15,469 GiB |
| CPU / RAM | 20 mantıksal CPU / yaklaşık 64 GB RAM |
| Paylaşılan yük | PitchServer dahil 166 çalışan konteyner |
| Compose / servis | `pitchserver` / `tribe` |
| Konteyner | `pitchserver_tribe` |
| Mevcut erişim | `127.0.0.1:18090`, `https://pitchserver.machinity.ai` |
| Son dağıtılan imaj; gerçek TRIBE aday seçimi dahil | `sha256:942421b4faa73f9788f61e1d75e704ef3aac84c9e399ade79a709fc42e439eb0` |
| Önceki sağlamlık / performans imajı | `sha256:9572265cf67a3be3186ff479552126aa370f06deb93240a7563470d010e50721` |

Traefik, diğer Compose projeleri, GPU sürücüsü ve sistem paketleri değiştirilmedi. İşlem öncesinde yaklaşık 4 GiB GPU belleği başka süreçlerce kullanılıyordu. Sunucunun swap alanı doluydu; PitchServer'ın swap kullanması engellendi.

## Akış ve bulunan sorunlar

Sunucuya dağıtılan imaj Python backend'idir. Masaüstü ve web değişiklikleri bu branch üzerinde build/test edildi; hazır native geliştirme build'i `src-tauri/target/debug/pitchcheck-desktop` yolundadır.

Masaüstü Tauri istemcisi SSH tüneli ve kullanıcı oturumu üzerinden; mobil istemci HTTP API üzerinden çalışıyor. Python API metni kelime olaylarına çeviriyor, Hermes metin özelliklerini çıkarıyor, TRIBE ile 20.484 voxel tahmini oluşturuyor, sayısal özellikleri özetliyor ve mevcut OpenRouter modeliyle raporu tamamlıyor. Doğrudan metin modunda TTS/WhisperX çalışmıyor. Son takip çalışmasında `/refine`, üç farklı LLM taslağını ve orijinali gerçek TRIBE ile ölçer; olgusal/bağlamsal kontrolü geçen ölçülmüş adaylar arasından mevcut nöral/semantik ağırlıklarla seçim yapar. Seçimden sonra ölçülmemiş yeni bir metin yazılmaz.

| Sorun | Düzeltilen davranış ve kanıt |
|---|---|
| İstemci isteği iptal edilince GPU kilidi erken bırakılıyordu | İşçi kendi izin hakkını taşıyor. HTTP iptali veya zaman aşımı sonrasında çalışan thread tamamlanana kadar yeni GPU işi ve model boşaltma engelleniyor. İptal ve timeout regresyonları geçti. |
| LLM çağrıları sınırsız thread işi oluşturabiliyordu | Skor yorumlama ve refine ortak iki işlem sınırını kullanıyor. Kuyruk 5 saniyede `429`, istemciye yönelik LLM süresi 180 saniyede `504` veriyor. İşçi arka planda sürerse izin hakkı korunuyor. |
| Doğrudan API ve web refine için parçalı büyük gövdeler sınırsız okunabiliyordu | Python API ve iki Next.js rotası 128 KiB gerçek byte sınırını uyguluyor. Gövde okuma süresi 15 saniye. Parçalı büyük istek `413`, duran gövde `408` testleri geçti. |
| Giriş/parola doğrulama kilitleri event loop'u durdurabiliyordu | Parola işlemleri ve auth dependency framework thread havuzunda çalışıyor. Skor sırasında sağlık yanıtları canlı ölçüldü. |
| Oturum tablosu giriş yapıldıkça büyüyordu | Süresi geçmiş oturumlar yeni girişte temizleniyor; en fazla 128 oturum tutuluyor. Limit dolunca en eski oturum çıkarılıyor. |
| Üst kütüphanenin sardığı CUDA OOM hatası fallback'i atlıyordu | Hata nedenleri zinciri inceleniyor. Sarmalanmış OOM için Accelerate/CPU kurtarma regresyonu geçti. Accelerate bütçesi toplam GPU belleği yerine kullanılabilir bellekten hesaplanıyor. |
| Metin modeli her yeni metinde yeniden yükleniyordu | TRIBE'ın üst kütüphanedeki koşulsuz encoder boşaltması, mevcut `TRIBE_UNLOAD_TEXT_MODEL_AFTER_SCORE` politikasıyla uyumlu hale getirildi. Encoder işlem aralarında korunuyor. |
| Her kelime için bütün bağlamın bütün katmanları CPU'ya kopyalanıyordu | Yalnızca hedef kelime token'ları CPU'ya aktarılıyor. Token/katman birleştirme aritmetiği aynı. Gerçek Torch testi padding ve çok token'lı hedefleri kapsıyor. |
| Kelime özellikleri RAM ve diskte birikiyordu | Kütüphanenin kendi feature-cache temizleme işlemi her tahminin `finally` bloğunda çalışıyor. Model ağırlıkları korunuyor; sınırlı tahmin önbelleği kullanılıyor. GPU karşılaştırmasında her işlemden sonra feature-cache kayıt sayısı sıfır. |
| Checkpoint'teki eğitim ayarı 20 veri yükleyici süreci açıyordu | İnferans `num_workers=0` kullanıyor; konteynerde gereksiz süreç ve bellek kopyaları oluşmuyor. |
| Offline modda hazır ağırlıklar olmasına rağmen model adı doğrulaması internete gidiyordu | Yalnızca yerelde config dosyası bulunan model adı offline doğrulamada kabul ediliyor. Offline, salt okunur ağırlık mount'u ile soğuk inferans geçti. |
| Hata logları müşteri metni veya sağlayıcı hata gövdesi içerebiliyordu | Loglar hata sınıfı / HTTP durumuyla sınırlı. Gizli metin içeren hata regresyonunda müşteri içeriği loga düşmüyor. |
| Masaüstü bağlantısı sunucu Compose/env dosyalarını yazıyor ve 15 dakikalık `latest` güncellemesi kuruyordu | Bağlantı mevcut dağıtımı kullanıyor; durmuşsa yalnızca `tribe` servisini yerel imajla başlatıyor. Bağlantı dosya veya cron yazmıyor. Başarısız girişte SSH tüneli kapatılıyor. Sunucudaki eski otomatik güncelleme kaldırıldı. |
| Web health backend kapalıyken başarılı görünüyordu | GET health backend kapalıyken `ok:false` ve `503` döndürüyor. HEAD hafif web liveness kontrolü olarak korunuyor. |

## Kaynak ve işletim sınırları

| Kaynak / koruma | Uygulanan değer |
|---|---|
| CPU | 4 CPU kota; OMP/MKL/OpenBLAS için 2 thread |
| RAM | 12 GiB konteyner limiti |
| Swap | 0 ilave swap |
| GPU | PyTorch allocator için 8 GiB; GPU `0` |
| Süreç | 256 PID |
| Paralel GPU işi | 1 |
| Paralel LLM işi | 2 |
| Tahmin önbelleği | En fazla 8 kayıt / toplam 64 MiB |
| Kullanılmayan model | 600 saniye sonra boşaltma |
| Log | 10 MB × 3 dosya döndürme |
| Kapanış | İşçiler tamamlanmadan model boşaltılmıyor; konteyner durma bütçesi 40 saniye |
| Süreç ayrıcalıkları | `no-new-privileges`; tüm capability'ler kaldırılmış, yalnızca karışık sahiplikteki model/auth dosyaları için `DAC_OVERRIDE`, `FOWNER` korunmuş |
| Dağıtım | Yerel imaj ID'si sabit, `pull_policy: never`, `--no-deps tribe` |

Ortak üst dizinin `.env` dosyası artık servise aktarılmıyor. Mevcut OpenRouter ve auth ayarları ayrı `runtime.env` dosyasında, `0600` izinleriyle korunuyor. Mevcut auth dosyası ve kullanıcı parolası değiştirilmedi. Sunucunun Compose dosyası `0444`; eski masaüstü sürümlerinin bağlantı sırasında dosyayı yanlışlıkla yeniden yazması engelleniyor. Yeni masaüstü kodu bu dosyaya yazmadan bağlanır; eski provisioning yapan sürümler güncellenmelidir.

## Ölçülmüş hız

Eski imaj ve son kod, aynı RTX 5080 üzerinde sırayla; 4 CPU, 12 GiB RAM, 8 GiB allocator bütçesi, batch=4 ve boş ayrı feature dizinleriyle çalıştırıldı. Hugging Face ağırlıkları diskten ve offline yüklendi. Exca'nın rastgele feature iş sırası için karşılaştırma seed'i sabitlendi. Metinler sentetik; müşteri verisi kullanılmadı. Son temizleme düzeltmesi dahil ölçümler:

| Girdi | Eski imaj | Yeni kod | Azalma |
|---|---:|---:|---:|
| 42 kelime; model soğuk | 4,8947 sn | 4,2963 sn | %12,2 |
| 126 kelime; farklı yeni metin, model sıcak | 4,6019 sn | 2,3823 sn | %48,2 |
| 378 kelime; farklı yeni metin, model sıcak | 24,7823 sn | 15,4330 sn | %37,7 |
| Aynı 126 kelimelik metin; tahmin önbelleği | 1,6 ms | 0,8 ms | GPU tekrar çalışmıyor |

Üç boyut için tüm voxel tahmin matrislerinde maksimum mutlak fark **0,0**. GPU'da ölçülen tepe PyTorch allocation **7,022 GiB**; yeni süreç için ölçülen tepe RSS yaklaşık **5.965 MiB**. Bunlar bu iş yükünün ölçümleridir; tüm metinler için performans garantisi veya birden fazla tekrarın istatistiksel medyanı değildir.

Batch=8 denemesi aynı sayısal tahminleri korumadı; kullanılmadı. Exca eksik feature işlerini rastgele sıraya koyduğu için soğuk feature-cache ve farklı batch bileşimleri önceki sürümde de farklı sayısal çıktılar oluşturabiliyor. Parite testi aynı sıralama/seed altında yapılır; LLM yanıtlarının deterministik olduğu iddia edilmez.

## ONNX / TensorRT kararı

Bu dağıtım mevcut PyTorch CUDA akışını kullanıyor. Ölçülen darboğazlar gereksiz encoder yüklemesi ve tam bağlamın CPU'ya taşınmasıydı; mevcut ağırlık ve aritmetik korunarak giderildi. İlk dağıtımda ONNX/TensorRT dönüşümü uygulanmadı.

Takip çalışmasında 28 encoder bloğu TensorRT'ye dönüştürülüp gerçek TRIBE tahminleri üretildi. Sıcak encoder ile 126 kelimede native **2,3397 sn**, TensorRT **2,4836 sn** ölçüldü; çıktı eşitliği testi geçmedi. 378 kelimelik TensorRT işi 8 GiB süreç GPU bütçesini aşınca yalnızca deneme konteyneri durduruldu. Canlı imaj korunuyor. Ayrıntılar, kaynak sınırları ve tekrar üretme yolu [TensorRT deneme raporunda](tensorrt-experiment.md).

Ayrıca tam API süresinde dış LLM çağrısı önemli yer tutuyor: ilk sağlamlık dağıtımında soğuk skor+rapor **17,060 sn**, refine **19,939 sn** ölçüldü. O imajın otomatik restart testi sonrası **18,549 sn** ve **15,495 sn** ölçüldü. Bu ilk ölçümde mevcut `google/gemini-3.8-flash` ve refine kalite adımları korundu. Encoder'ın TensorRT'ye taşınması bu dış çağrının süresini azaltmaz. TensorRT seçilirse dinamik şekiller için profil ve bellek bütçelerinin yeniden doğrulanması gerekir; bu NVIDIA'nın [dinamik şekil dokümanında](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-basics.html) anlatılır.

## İlk sağlamlık dağıtımının doğrulaması ve kapsamı

- Sabit GPU imajında **130 Python testi geçti**. Hafif CPU/CI ortamında **129 geçti, 1 Torch/neuralset testi atlandı**; o test gerçek GPU bağımlılıklarının bulunduğu imajda geçti.
- **49 web testi**, **9 Rust testi**, ESLint, Next.js üretim build'i, desktop static build ve native desktop build geçti.
- Python mock testleri CI'ye eklendi. Test bağımlılıkları sunucuda doğrulanan sürümlere sabitlendi. Paylaşılan sunucu Compose dosyasına ayrı CI config kontrolü eklendi.
- Gerçek API: kullanıcı girişi, auth olmadan `401`, büyük istek `413`, geçersiz istek `422`, gerçek TRIBE skoru, gerçek sağlayıcıyla refine, işlem sırasında unload engeli ve işlem sonunda GPU boşaltma doğrulandı.
- İzole testte işlem sırasında en uzun health yanıtı **102 ms**; canlı ilk geçişte **469 ms**, son imajda **121 ms**, son imajın restart testi sonrası **457 ms** oldu.
- Yalnızca PitchServer'ın ana sürecine SIGTERM gönderilerek `unless-stopped` otomatik restart davranışı doğrulandı. Kalıcı auth dosyasıyla tekrar giriş, gerçek skor ve refine yeniden geçti. Sunucuya reboot uygulanmadı.
- Yeni masaüstü bağlantısının ürettiği SSH komutu canlı hostta çalıştırıldı; mevcut sabit imajı döndürdü ve dosya yazmadı. GUI üzerinden SSH/parola bağlantısı bu çalışmada ayrıca tıklanarak test edilmedi.
- Public HTTPS health yönetilen Firefox'ta görüntülendi; auth gerekli, model CUDA, bellek sınırı 8 GiB ve Rust sayısal çekirdek aktif görüldü. Görev tarayıcı oturumu kapatıldı.
- Başlangıç kaydıyla karşılaştırmada **diğer 165 servisin tamamı** aynı konteyner kimliği, başlangıç zamanı, restart sayısı ve sağlık durumuyla kaldı. Test konteynerleri kaldırıldı.

Kod incelemesi `ce-code-review` ve sadeleştirme kontrolü `ce-simplify-code` ölçütleriyle ana oturumda sırayla yapıldı. Kullanıcının AGENTS talimatı nedeniyle bağımsız alt ajan veya ikinci model incelemesi yapılmadı. İnceleme tamamlandı; kaynak, API sözleşmeleri, iptal/kapanış davranışı, auth, hata logları, cache ömrü ve dağıtımın tek servise sınırlandırılması kontrol edildi.

8 GiB sınırı PyTorch allocator'a aittir; bu GPU'da donanımsal ayrı GPU bölümü veya bağımsız GPU zamanlayıcı kotası değildir. Diğer GPU işlerinin gecikmesine sıfır etki garantisi verilmez. En büyük 30.000 karakterlik giriş ve aynı anda yoğun başka GPU işleri paylaşılan canlı makinede stres testine sokulmadı. In-memory auth oturumları restart'ta kaybolur; kullanıcı yeniden giriş yapar. Bu sınırlar çalışan imajın ve gerçek testlerin kapsamıdır.

## Tekrar çalıştırma ve geri dönüş

Yerel kontroller:

```bash
python -m pip install -r tribe_service/requirements-test.txt
python -m pytest tribe_service/tests -q
npm ci
npm run lint
npm test
npm run build
npm run build:desktop-web
cargo test --manifest-path src-tauri/Cargo.toml
cargo build --manifest-path src-tauri/Cargo.toml
```

Sunucudaki mevcut sürümü kontrollü yeniden başlatmak/başlatmak:

```bash
ssh -p 2022 dkmserver@78.186.120.189
cd /home/dkmserver/Desktop/Machinity/aytug/pitchserver
./update-pitchserver.sh
```

Bu komut imaj çekmez; yalnızca sabit mevcut `tribe` servisini uzlaştırır. Yeni bir sürüm için ayrı imaj oluşturulup test edilmeli, `.env` içindeki `PITCHSERVER_IMAGE` doğrulanan yeni yerel imaj ID'siyle değiştirilmelidir. Model/auth mount'ları ve shared Traefik korunmalıdır. `Dockerfile.server` doğrulanmış temel GPU imajına uygulama kodunu kopyalar; sürücü veya tüm bağımlılıkları tekrar kurmaz.

Son değişiklik öncesine dönüş:

```bash
/home/dkmserver/Desktop/Machinity/aytug/pitchserver/rollback-20261004T135434Z/rollback.sh
```

Çalışma başlamadan önceki orijinal servise dönüş:

```bash
/home/dkmserver/Desktop/Machinity/aytug/pitchserver/rollback-20261004T134614Z/rollback.sh
```

Rollback scriptleri yerel eski imajı ve kaydedilmiş Compose/env ayarlarını kullanır; yalnızca `tribe` servisini yeniler. Eski otomatik `latest` cron işi yeniden etkinleştirilmez. Yedeklerde sırlar bulunduğundan dizinler `0700`, dosyalar `0600` olarak sunucuda tutulur ve Git'e alınmaz.

Tekrar üretilebilir GPU karşılaştırması için `scripts/benchmark-runtime.py`, auth/API kabul kontrolü için `scripts/smoke-runtime.py` kullanılır. Karşılaştırma önce ayrı bir sınırlı konteynerde eski imajla `--output /audit/before`, ardından yeni imajla boş başka `TRIBE_CACHE_DIR` ve `--output /audit/after --compare /audit/before` ile sırayla çalıştırılmalıdır. Her iki koşuda aynı GPU/thread limitleri ve ortak read-only Hugging Face ağırlık mount'u kullanılmalıdır. `--compare` matris eşitliğini ve yeni sürümün işlem sonunda feature-cache'inin boş olmasını doğrular.

Sunucudaki ayrıntılı test çıktıları ve `.npy` matrisleri `audit-20261004/` altında; son performans kaydı `final/metrics.json`, servis karşılaştırması `service-verification.json`, dağıtım/rollback yolu `deployment.json` dosyalarında bulunur. Auth/API testi yalnızca durum, süre ve model adını kaydeder; parola veya token yazdırmaz.

## Taskfulight özel iOS entegrasyonu — önceki dağıtımın ölçümleri

Taskfulight'ın mevcut `feat/taskfulight-v2` branch'ine **Şimdi → İkna** ekranı eklendi. Ekran mevcut PitchServer hesabıyla HTTPS üzerinden bağlanır; giriş bilgileri ve token iPhone'un cihazda kalan, kilit açıkken erişilebilir Keychain kaydındadır. Kullanıcı bir dokunuşla modeli yükleyebilir veya boşaltabilir; analiz, iki turlu metin iyileştirme ve yeni metni yeniden analiz ederek karşılaştırma aynı ekrandadır. Seçilen model uygulamanın mevcut ayarından alınır: bu kabul testinde **`deepseek/deepseek-v4-flash`** kullanıldı. Sunucunun başka istemciler için varsayılan modeli değiştirilmedi.

Backend'de yeni auth gerektiren `POST /runtime/load`, TRIBE ile aynı cached Hermes encoder'ını tam yükler; sadece wrapper oluşturup hazır görünmez. Tekrar yükleme idempotenttir. Mevcut GPU işlem kilidi ve backpressure sınırları korunur. Unload isteği iptal edilse bile boşaltma thread'i bitene kadar pipeline kilidi tutulur. Yeni `POST /auth/logout` yalnızca arayan oturumu iptal eder; diğer oturumları korur.

Canlı testte DeepSeek'in varsayılan yüksek düşünme süresi public HTTPS sınırını aştı ve `524` oluştu. Yalnızca bu model için, operatör açık bir reasoning effort belirtmediyse, OpenRouter payload'ına `reasoning: {enabled: false}` eklenir. Hem skor raporu hem refine ortak politikayı kullanır. İkinci eleştirmen turu korunur. `minimal` denemesi aynı rapor tanılamasında **98,476 sn**, düşünme kapalı deneme **34,531 sn** sürdü; yalnız istemci timeout'unu artırmak edge timeout'unu çözmedi. Eksik `persuasion_score` alanıyla gelen JSON artık mevcut retry bütçesini kullanır; sessizce başarılı semantik rapor sayılmaz. Neural-only fallback uygulamada açıkça gösterilir.

Son imajda gerçek public HTTPS kabul sonucu; sentetik endüstriyel satış metni:

| İşlem / ölçüm | Sonuç |
|---|---:|
| Tam model yükleme | 4,783 sn |
| Tekrar yükleme | 0,124 sn |
| İlk analiz + gerçek DeepSeek raporu | 43,124 sn; skor 35 |
| Gerçek iki turlu refine | 9,528 sn |
| Yeni metnin ayrı gerçek analizi | 28,852 sn; skor 54 |
| Son native TRIBE hesabı; 27 kelime | 0,411 sn |
| İşlem sırasında en uzun health | 0,416 sn |
| Model boşaltma | 0,653 sn |
| Son analizde CUDA allocated / peak | 6,652 / 6,702 GiB |
| Model yüklü, işlem başlamadan RAM | 1,769 GiB |
| Model boşaltılmış RAM | 1,616 GiB |

RAM satırları Docker working-set ölçümüdür; yalnız yükleme/bekleme durumunu kapsar. Daha uzun gerçek inference için önceki performans karşılaştırmasında ölçülen tepe RSS yaklaşık 5.965 MiB idi. CUDA ölçümü GPU belleğidir ve RAM'den ayrıdır. Skor artışı bu tek örnekteki model değerlendirmesidir; gerçek kişilerin ikna olma olasılığı veya genel kalite garantisi değildir. Tam uygulama yolu fiziksel iPhone'da henüz doğrulanmadı; Flutter widget testleri ile native iOS/TestFlight CI ayrı kontrollerdir.

Son imajda **136 Python testi geçti**; yerel hafif ortamda 135 geçti, Torch/neuralset bağımlılık testi atlandı ve gerçek GPU imajında geçti. Auth olmadan load `401`, çalışan GPU işinde unload engeli, tam modelin sağlıkta hazır görünmesi, gerçek sağlayıcıyla iki turlu refine, gerçek yeniden skor, unload ve logout sonrası eski token'ın `401` alması doğrulandı. Public HTTPS health yönetilen Firefox'ta tekrar doğrulandı; görev oturumları kapatıldı. Test sonunda model boşaltıldı.

Diğer **165 kalıcı servisin** kimliği, başlangıç zamanı, restart sayısı ve sağlık durumu ilk Taskfulight snapshot'ıyla karşılaştırıldığında değişmedi. Son dağıtımın ham snapshot'ına eşzamanlı çalışan geçici pytest konteyneri de girmişti; `--rm` ile normal çıkışı ham karşılaştırmada bir silinme olarak görünür. Ham kayıt korundu; kalıcı servislerin ilk baseline'a göre ayrı doğrulaması `audit-taskfulight-20261004-final/service-verification.json`, açıklama `deployment-verification-note.json` dosyasındadır. GPU zamanında sıfır etki garantisi verilmez; mevcut kaynak sınırları korunur.

Kanıtlar sunucuda `audit-taskfulight-20261004-final/` altında `live-api.json`, `live-api.log`, `memory.json`, `deployment.json` ve `service-verification.json` dosyalarındadır. Sentetik test metni kayıtlıdır; parola/token/API anahtarı kayıtlı değildir. Son küçük backend değişikliği öncesine dönüş `rollback-20261004T180820Z/rollback.sh`; Taskfulight entegrasyonunun tümünü geri alıp önceki sağlamlık imajına dönüş `rollback-20261004T174806Z/rollback.sh` ile yapılır. Her iki script yalnız `tribe` servisini yeniler.


## Gerçek TRIBE aday araması — kullanıcı örneği sonrası son dağıtım

Önceki mobil refine yalnız dil modeliyle iki tur yazıyordu. Kullanıcının kişisel davet örneği, açıkça belirtilen hoşlanmama tercihinin göz ardı edildiğini ve verilmemiş kısa klip ayrıntısının yazıya eklendiğini gösterdi. Son akışta mevcut `deepseek/deepseek-v4-flash` üç farklı taslak üretir; orijinal dahil dört metnin her biri gerçek CUDA TRIBE modeliyle ölçülür. Eleştirmen her adayı doğruluk, asıl amaç, alıcı tercihi ve gönderenin dili açısından değerlendirir. Python, uygun adaylar arasında gerçek nöral kanıt ile semantik bağlamın mevcut ağırlıklı bileşimini kullanarak seçim yapar. Yüksek nöral puan yanlış bilgiyi meşrulaştıramaz; uygun iyileşme yoksa orijinal korunur.

Analizdeki örnek yeniden yazımlar artık taslak üretimine aktarılmaz; bunlar yeni gerçekler için kaynak değildir. Gerçek HTTPS ilk kabul denemesinde eleştirmenin önerideki verilmemiş cuma bilgisini onaylaması yakalandı. Tarih, sayı, klip/bilet ayrıntıları yalnız orijinal, persona ve gerçekten cevaplanmış bağlamda varsa kabul edilir. Sorulmuş ama cevaplanmamış soru olgusal kaynak sayılmaz. Doldurulmamış köşeli parantezli yer tutucular da elenir. Bu sınırlı Türkçe/İngilizce ayrıntı denetimi genel olgusal çıkarımın yerini tutmaz; kalan iddialar için semantik kontrol korunur. Başarısız veya eksik eleştirmen çıktısı ölçümsüz yeniden yazıma düşmek yerine redakte edilmiş hata verir.

GPU izin hakkı dört metnin bütün ölçüm süresi boyunca işçide kalır; iptal veya HTTP timeout bu hakkı erken bırakmaz. İşlem sırasında unload reddedilir. API yanıtındaki `tribe_guidance` orijinal ve üç adayın metnini, gerçek model/mode kimliğini, voxel/segment sayısını, nöral sinyallerini, ayrı bağlam puanını, seçimi ve elenme nedenlerini taşır. Taskfulight bu sözleşmeyi doğrulamadan gerçek TRIBE karşılaştırması iddiası göstermez. Uygulamadaki açılır aday karşılaştırması seçimin kanıtını görünür kılar; genel kanal kişisel davetlerde başlangıç seçimidir.

Eleştirmene gönderilen yinelenen eksen açıklamaları ve araştırma eki kaldırıldı; gerçek sayısal ölçümler korunuyor, tam eksen bilgisi API kanıtında kalıyor. Önceki ara sürümde dış sağlayıcı nedeniyle refine 104,159 sn sürdü. Bu bir hız garantisi değildir; gerçek model hesabı ile dış sağlayıcı gecikmesi ayrı ölçülür.

Son backend kaynak commit'i `52c2c7924159cb740fbd459136967f444c49b569`; `/app` altındaki 28 izlenen runtime dosyasının SHA-256 değeri kaynakla bire bir doğrulandı. GPU bağımlılık imajında **140 Python testi**, hafif yerel ortamda **139 test** geçti; yalnız gerçek Torch/neuralset testi yerelde atlandı ve imajda geçti. Karşıt testte aynı semantik değerlendirme tutulup nöral ölçümler değiştirilince kazanan aday değişiyor. Verilmemiş tarih/klip/bilet ve yer tutucu en yüksek model puanını alsa ve eleştirmen onaylasa bile eleniyor. İptal, eksik/çift aday, eksik eleştirmen ve özel hata içeriği regresyonları geçti.

Taskfulight kaynak commit'i `a85dcbb9c126bdc0120c834edf7d56573e90b350`. **136 Flutter testi**, analiz ve release hygiene geçti. Yeni imzalı iOS CI'de **12 native RunnerTests ve 2 simülatör akışı** geçti. **1.0.0 (1791142457)** Apple tarafından işlendi ve mevcut **Taskfulight Owner** grubuna eklendi; dış beta/App Store yayını yapılmadı. IPA bundle ID `com.aytzey.taskfulight`, kaynak provenance ve SHA-256 (`83cd4e7eb9ed93d9399214c68d65b7a7c5230e62b9e930b1d704a95bcdc2d5e1`) CI artefaktlarıyla eşleşiyor. Yeni sürümün fiziksel iPhone akışı bu çalışmada çalıştırılmadı.

TRIBE, [resmî proje açıklamasındaki](https://github.com/facebookresearch/tribev2) ortalama-denek fMRI yanıtını tahmin eder. Bu kullanıcının flörtünden ölçülmüş fMRI veya ikna olma olasılığı değildir; türetilmiş nöral eksen/puanlar bireysel davranışa doğrulanmış kalibrasyon sayılmaz.

Son dağıtım yalnız `tribe` servisini yeniledi. Diğer **165 konteynerin** kimliği, başlangıç zamanı, restart sayısı ve sağlık durumu değişmedi. Kaynak sınırları, auth/model mount'ları, Traefik ve native PyTorch CUDA korunuyor. Son sürüm öncesine hedefli dönüş `rollback-20261004T194256Z/rollback.sh`; bu aday seçimi çalışmasının tamamından önceki imaja dönüş `rollback-20261004T192626Z/rollback.sh`. Sunucudaki kanıt dizini `audit-tribe-search-final-20261004/`; kullanıcıya ait örnek metinler sadece yetkisi sınırlı canlı kabul kaydında tutulur ve Git/PR açıklamasına yazılmaz.

Son public HTTPS kabulünde kullanıcı girdisine karşı **4 gerçek, mock olmayan TRIBE ölçümü** (20.484 voxel) doğrulandı. Nöral türetilmiş puan orijinalde **32,357**, seçilende **38,637**; ayrı semantik bağlam puanı **42,5 → 60,5**, rapor puanı **25 → 40** oldu. Bunlar aynı kişinin gerçek davranışını ölçmez; yalnız bu işlemdeki model değerlendirmeleridir. Seçilen metin kısa ve verilen bilgilere sadık kaldı; doğal dil kalitesi hâlâ model değerlendirmesidir, evrensel ikna iddiası yoktur.

| Son canlı kabul ölçümü | Sonuç |
|---|---:|
| Tam yükleme | 4,783 sn |
| Orijinal skor + rapor | 24,807 sn |
| Üç aday üretimi + dört ölçüm + eleştirmen/seçim | 74,633 sn |
| Ayrı yeniden skor + rapor | 17,715 sn |
| Yeni adayların tekil native TRIBE hesabı | 0,196–0,240 sn |
| Health en uzun | 0,340 sn |
| 150 ms aralıklarla örneklenen cgroup `memory.current` tepe | 1,617 GiB |
| Örnek CUDA peak allocated | 6,695 GiB |
| Unload | 0,607 sn |

Bu kısa metin denemesinde dört TRIBE ölçümü yaklaşık bir saniyedir; toplam refine süresinin çoğu dış LLM beklemesidir. **74,633 saniye hızlı bir son kullanıcı yanıtı sayılmaz**; seçili ucuz sağlayıcının gecikmesi değişkendir. Yerelde aynı modelle aday üretimi yaklaşık beş saniye sürerken bu sunucu çağrısında belirgin gecikti. Bu örnekleme mutlak süreç RSS veya RAM tepe ölçümü değildir. En büyük metinler için RAM veya süre garantisi çıkarılmaz. Sıfır aktif GPU işi, başarılı unload, çağıran token'ın logout sonrası `401` alması ve diğer 165 servisin son kabul sonrasında da değişmemesi doğrulandı.

## Deneysel yapısal varsayım v2 — kaynak değişikliği

Kullanıcının talebiyle mevcut Hermes encoder korunur ve ölçülen model yanıt geometrisi varsayım olarak kullanılmaya devam eder. TRIBE checkpoint'inin beklediği encoder `meta-llama/Llama-3.2-3B`, mevcut encoder `NousResearch/Hermes-3-Llama-3.2-3B` olduğu için fizyolojik genelleme doğrulanmış sayılmaz. Bu uyumsuzluk sayısal gözlemleri gizleyen bir veto değildir; API, native/Python yollarında gerçek ve beklenen encoder kimliklerini açıkça taşır. Mutlak büyüklük, anatomik ROI, duygu veya ikna olasılığı olarak yorumlanmaz.

Her iz kendi ortalamasına bölünür. İlk/son çeyrek ve yarım pencerelerde giriş/kapanış oranları ile gruplanmış ardışık düşüşler gözlenir. Orijinalin en büyük yapısal açığı giriş, süreklilik veya kapanış deneyini belirler; aynı hedef Jev'in yazım öncesi kararına, yaklaşık kelime sırası hedeflerine ve bütün aday karşılaştırmalarına gider. Sayısal ağırlık mevcut çıktı-sağlığı ağırlığı × `min(1, (segment−1)/4)` × `min(1, iz_aralığı/0.0005)` olarak hesaplanır; son çarpan dört ondalıkta yuvarlanan izde küçük farkları azaltır. Aday etkisi aynı hedefin pencere değişimlerinin ortalamasıdır; her değişim −1…1 aralığında sınırlandırılır. Karşılaştırma ağırlığı orijinal/aday sayısal ağırlıklarının küçüğü × orijinal hedefin pencere desteği × yön uyumu × `min(1, ortak_pencere_sayısı/2)` olur. Farklı uzunluklar oransal pencerelerle karşılaştırılır; min/max değişim ve yön uyuşmazlığı duyarlılık olarak gösterilir, istatistiksel güven aralığı değildir. Ham uzamsal özetler ve farkları ayrı gözlemlerdir.

Doğruluk, asıl amaç, alıcıya saygı ve üslup kontrolleri önce uygulanır. En iyi bağlam puanının üç puan yakınındaki uygun adaylarda `seçim = clamp(bağlam + 3 × karşılaştırma_ağırlığı × etki, 0, 100)` kullanılır; gösterilen ve sıralanan puan aynıdır. Sıfır/sabit/mock çıktının ölçülebilir zamansal etkisi yoktur; zayıf ama değişken gerçek çıktı küçük, görünür katkı sağlar. Jev'in serbest kazanan tercihi bu politikayı geçersiz kılamaz. Opt-in akış tek `google/gemini-3.5-flash-lite` isteğinde üç aday üretir; düşük gizli reasoning ve toplam 1.536 token sınırı vardır. Yeniden yazım döngüsü, pahalı model veya sağlayıcı geri dönüşü yoktur. Bu bölüm kaynak politikasını açıklar; yeni canlı kabul sonuçları ayrıca kaydedilir.

İlk v2 canlı kabulünde gerçek Hermes/native yol, görünür deneysel katkı ve altyapı kontrolleri geçti: soğuk public refine 8,734 sn, yazar isteği 1,142 sn ve $0,000794574; sağlayıcı 0 reasoning token bildirdi. Native imajda 161 Python, mobilde 136 Flutter testi geçti; yükleme/refine/unload sonrasında diğer 165 servis değişmedi. GPU başlangıç düzeyine döndü; konteyner belleği unload sonrasında yaklaşık 1,60 GiB kaldı, CPU belleğinin tamamı serbest kalmış değildir. Dil kalitesi kabulü geçmedi: bir aday verilmemiş zaman ekledi ve elendi; kalan yazılar seçilen oyuncu üslubuna rağmen genel taviz/rica kalıbında kaldı. Sayısal ve altyapı başarısı iyi ikna metni kanıtı sayılmaz.

Bu soruna yönelik sınırlı düzeltme, aynı tek ucuz istekte girdiden alınmış dayanak, kısa somut fikir ve tam mesaj içeren üç nesne üretir. Yazar yalnız bağlamı, Jev'in uygulanacak hamlelerini ve seçilen yapısal hedefin gerçek pencere gözlemleri/yaklaşık hedefini alır; yinelenen tam ölçümler ve uzun örnekler kaldırılmıştır. İkinci aday tamamlayıcı bir hamle, üçüncü aday yalın davet kullanır. Sunucu dayanağı doğrular, mesajı parçalamadan korur; hamle/hedef kimliklerini doğrulanmış Jev kararından ekler. Yazım seçimleri mevcut `writer_call` kanıtında görünür; gizli muhakeme dökümü değildir. Seçim puanları ve deneysel formüller değişmez. İlk ayrı taslak+kopya sözleşmesi HTTP 200 yanıtına rağmen doğrulamada başarısız oldu; yanıt kaydedilmediği için tam nedeni bilinmiyor. Sonraki açılış/devam denemesi parse edildi ancak noktalama ve kalıp rica sorunları nedeniyle kaliteyi geçmedi. Sınırlı deneme yalnız `message.content` ve güvenli hata metnini kaydeder, reasoning alanlarını kaydetmez. Yeni GPU ölçümü olmadan yazar denemesi yeni adayların fMRI doğrulaması sayılmaz; somut dil kalitesi ayrıca değerlendirilir.

Kısa fikir+mesaj denemesi de kaliteyi geçmedi: ilk aday soru yerine hoş bir bildirimdi; ikinci aday "acayip eğleneceğiz" vaadi ve "gelmelisin" emri kullandı, üçüncü aday da emri korudu. Yazar isteği 726 girdi/219 çıktı token, 1,901 sn ve $0,0007653 idi; parse başarısı ikna kalitesi olarak sunulmaz. Son sınırlı düzeltme Türkçe kaynak için sistem önceliğinde doğal davet sorusu, zevki değiştirmeden ortak etkinlik fikri ve farklı etkinlikten tek olumlu örnek verir. Şema, ucuz model, tek çağrı ve ölçüm/seçim politikası korunur; sonuç son tam public kabulde ayrıca değerlendirilir.

`61dd398` kaynağının tam public kabulünde soğuk refine **9,184 sn**, gerçek `google/gemini-3.5-flash-lite` yazım isteği **1,579 sn / $0,000906642** sürdü. Orijinal ve üç aday mevcut Hermes ile gerçek TRIBE/native yolunda **20.484 voxel** olarak ölçüldü; encoder eşleşmesinin deneysel sınırı korunur. Kaydedilen konteyner RAM yaklaşık **1,62 GiB**, CUDA allocated/peak **6,652 / 6,695 GiB** idi; unload doğrulandı, diğer **165 servis** değişmedi. Bu kabulde iki seçim hatası yakalandı: açık bir ortak an önerisi olgusal iddia sayılarak elendi; verilmemiş zamanın Türkçe çekimi denetimden kaçtı.

Sonraki **editör replay**, aynı kayıtlı dört metin ve aynı gerçek ölçümler üzerinde yalnız **bir Jev çağrısı** yaptı (**0,727 sn / $0,00037611**); yeni yazım veya GPU ölçümü yapılmadı. Düzeltilmiş ölçütle c1'in mecazi ortak an önerisi uygun bulunup seçildi; c2 baskı nedeniyle, c3 verilmemiş `bu akşamı` → `bu akşam` zamanı nedeniyle elendi. Seçim c3'ten c1'e değişti. c1'in bağlam puanı **47,062**, deneysel katkı sonrası seçim puanı **47,008** oldu: negatif ölçüm katkısı korunmuştur, nöral veya insani ikna artışı iddia edilmez. Bu kanıt yeni kodun tam uçtan uca public kabulü değildir. Yerel kontroller **168 geçti / 1 bağımlılık testi atlandı**; sonraki imaj/dağıtım kanıtı ayrıca kaydedilir.

TaskFlight `8a3fde4`, CI `6ac2d597ff057bd79546477f` üzerinden **1.0.0 (1791154203)** olarak Apple tarafından işlendi ve yalnız mevcut **Taskfulight Owner** TestFlight grubuna teslim edildi; kaynak ve özel grup teslimi doğrulandı. Bu editör düzeltmesinde mobil kod değiştirilmedi. Canlı ve replay artefaktları sırasıyla `pitchcheck-empirical-final-live.json` ve `pitchcheck-editor-replay-results.json` kayıtlarıdır.

## Encoder karşılaştırması ve özellik önbelleği düzeltmesi — 5 Ekim 2026

İlk gerçek, önbelleksiz A/B'de aynı metnin iki ölçümü arasında **0,35–0,74 bağıl L2 farkı** görüldü. CPU üzerinde neden yeniden üretildi: exca 0.5.20 önbelleği silerken eski dosyanın memmap'ini ve JSONL okuma konumlarını tutuyordu. Aynı dosya adı yeniden kullanılıp kelimelerin hesaplama sırası değişince 11/22 özellikleri 22/11 olarak okunuyordu. `f6b5cd9`, başarılı ve başarısız işlem temizliğinde yalnız ilgili özellik önbelleğinin okuma durumunu aynı klasör/politikayla yeniden başlatır. Ayrıca Hermes'in farklı pad/EOS kimlikleri yerine gerçek attention mask kullanılır; sol/sağ dolgu ve gerçek EOS korunur. HF ağırlık önbelleği, seed, decoder ve diğer özellik klasörleri değiştirilmez. Kurulu runtime'da **172 test geçti**.

Düzeltilmiş `cc75b22…` imajında iki encoder ayrı süreçlerde, aynı checkpoint/native CUDA/BF16, batch 4, iki işlemci thread'i, **4 CPU / 12 GiB RAM / 8 GiB CUDA** sınırlarıyla karşılaştırıldı. 42/126/378 kelimelik metinler ve kısa kişisel davet için ikişer gerçek, önbelleksiz ölçüm yapıldı. Soğuk yükleme yeni model sürecidir; HF dosyaları hazırdır, işletim sistemi sayfa önbelleği silinmemiştir. Süreler iki ölçümün medyanıdır; 42 kelime için ilk inference ısınması dahildir.

| Ölçüm | Hermes | Checkpoint'in Llama-3.2-3B encoder'ı |
|---|---:|---:|
| Soğuk yükleme | 4,022 sn | 4,066 sn |
| 42 / 126 / 378 kelime | 0,753 / 1,863 / 10,204 sn | 0,752 / 1,873 / 10,316 sn |
| Kısa davet | 0,315 sn | 0,306 sn |
| Süreç tepe RSS | 5,814 GiB | 5,815 GiB |
| CUDA peak allocated / reserved | 7,020 / 7,039 GiB | 7,201 / 7,287 GiB |
| Cgroup `memory.peak` | 1,805 GiB | 1,973 GiB |
| Aynı girdinin bağıl L2 farkı | 0,001183–0,005123 | 0,003429–0,006076 |
| En düşük tekrar korelasyonu | 0,9999837 | 0,9999786 |

Bütün matrisler sonlu ve 20.484 voxel içerir; fallback olmadı. Büyük yanlış-özellik sapması giderildi, fakat BF16 ve değişen batch bileşimiyle uyumlu küçük farklar kaldı: **`allclose(rtol=1e-4, atol=1e-5)` geçmedi**, bit düzeyinde eşitlik veya kusursuz kararlılık iddia edilmez. Cgroup ve RSS farklı ölçümlerdir; paylaşılan dosya önbelleğinin cgroup'a atfedilmesi nedeniyle düşük cgroup değeri RAM tüketiminin dramatik azaldığını kanıtlamaz. Önceki büyük sapmalı ölçümler bu koşuyla doğrulanmış sayılmaz. Bu A/B model davranışını karşılaştırır; ikna başarısı veya fizyolojik doğruluk deneyi değildir. Canonical encoder'ın gerekçesi checkpoint'in eğitildiği özellik eşleşmesidir; hız üstünlüğü iddiası yoktur.

Bu karşılaştırmada **yazar/Jev çağrısı ve API maliyeti sıfırdır**; ucuz tek yazım akışı ve mobil kaynak korunur. İki koşunun toplam bakım aralığı **68,086 sn** oldu; önceki canlı Hermes imajı tekrar başlatıldı, diğer **165 servisin** kimliği, başlangıcı, restart ve sağlık durumu değişmedi. Canonical canlı geçiş ve rollback kanıtı ayrı kaydedilir.

Son dağıtımda aynı test edilmiş **`sha256:cc75b22ee19f23163e58f654d734e15dfab009256726b375de430ffa24cad3b1`** imajı yalnız `tribe` için etkinleştirildi; Compose'un kullandığı gerçek `runtime.env` içinde encoder `meta-llama/Llama-3.2-3B` seçildi. Sağlık kontrolü geçti, diğer **165 servis** değişmedi ve ek konteyner kalmadı. Public HTTPS üzerinden gerçek auth ile ilk yükleme **4,764 sn**, idempotent tekrar yükleme **0,257 sn** sürdü; gerçek/beklenen encoder canonical, `text_feature_compatible=true`, CUDA ve batch 4 doğrulandı. Unload `200 / model_loaded=false`, logout `200` verdi; bu kontrol yeni skor veya yazar/Jev çağrısı içermez.

Son durumda model boşaltılmıştır. Host GPU belleği bütün iş yükleriyle **4.217 MiB kullanılan / 11.624 MiB boş**, GPU kullanım oranı `%0`; bizim konteynerin Docker working-set'i **836,8 MiB / 12 GiB** idi. Working-set süreç RSS değildir ve unload'un bütün CPU belleğini serbest bıraktığı anlamına gelmez.

Tekrar üretim betiği, iki encoder'ın ham `.npy` matrisleri, `metrics.json` ve bakım snapshot'ları sunucuda `/home/dkmserver/Desktop/Machinity/aytug/pitchserver/encoder-ab-20261005/run3-fixed/` altındadır. Koşullar aynı sınırlı, sıralı konteynerlerde iki tekrar, prediction/feature cache temizliği ve read-only HF ağırlıklarıdır; işletim sistemi önbelleği silinmez. Canlı dağıtım ve public yükleme kanıtları `/home/dkmserver/Desktop/Machinity/aytug/pitchserver/audit-encoder-stability-final-20261005-f6b5cd9/` altında `deployment.json` ve `canonical-public-load.json` dosyalarındadır. Önceki imaj ve Hermes ayarına hedefli dönüş `/home/dkmserver/Desktop/Machinity/aytug/pitchserver/rollback-encoder-stability-final-20261005-f6b5cd9-20261005T005719Z/rollback.sh` ile yapılır; bu script korumalı yedeklerden `.env` ve gerçek `runtime.env` dosyalarını geri getirip yalnız `tribe` servisini yeniler.

## GLM Flash geçişi — 5 Ekim 2026

Taskfulight İkna stüdyosunun yazım ve açık model seçimiyle analiz istekleri `z-ai/glm-5.3-flash` kullanır. Eski uygulamanın `jevStrategy: true` ile gönderdiği Gemini model kimliği de sunucuda aynı GLM yazara yönlenir. Jev strateji/editör, orijinal `meta-llama/Llama-3.2-3B` encoder, gerçek TRIBE ölçümleri ve deneysel seçim formülü korunmuştur. Uygulama, yanıttaki gerçekten kullanılan model kimliğini gösterir. Ürün kodunu Sol 6.1 xhigh yazdı; ebeveyn ajan diff, gerçek çıktı ve dağıtım kontrollerini yaptı.

Yazar tek uygulama isteğinde üç aday üretir; `reasoning.effort=low`, `exclude=true`, toplam completion sınırı **1.536 token**. `exclude` düşünmeyi kapatmaz; GLM katalogda zorunlu reasoning bildirir. Açık GLM analiz isteği low/4.096 token kullanır. Baseten FP8, Fireworks ve CoreWeave sırasıyla önceliklidir; kapasite yetmezse gerekli parametreleri destekleyen aynı model sağlayıcılarına geçiş açıktır. Yazarda uygulama seviyesinde tekrar veya yeniden üretim döngüsü yoktur.

İlk aday imajı sekiz isteği tamamladı, fakat altı istek yavaş sağlayıcılara düştü: tam refine medyanı **15,531 sn**, en uzunu **28,513 sn** oldu. Tutum tersine çevirme, verilmemiş geçmiş ve emir kopyalama örnekleri de görüldü. Yalnız hızlı sağlayıcılara izin veren sonraki aday, üçüncü istekte upstream `429` nedeniyle durdu. Ücretli anahtarda kullanılabilir kota vardı; Baseten, Fireworks ve CoreWeave ayrı küçük bağlantı denemelerinde de upstream kota hatası verdi. Bu nedenle hız tercihi korunurken aynı model kapasite yedekleri yeniden açıldı. Her başarısız aday kaldırıldı ve önceki imaj sağlıklı geri getirildi.

Son yazım talimatı gönderenin hevesini ve amacını korur, kaynak emri gönüllü davete dönüştürür; verilmemiş önceki istek/onay/tarih ve yer tutucu eklememesini ister. Jev, kaynaktan kopyalanmış veya arkasına soru eklenmiş emri de baskı olarak değerlendirir. Bunlar semantik model talimatlarıdır; deterministik dil kalitesi garantisi değildir.

Son imajın sekiz sentetik davet/iş örneği ve bir analiz isteği başarılıdır. Her refine tek yazar isteği, gerçek Jev ve orijinal dahil dört gerçek **20.484 voxel** TRIBE ölçümü içerir; encoder eşleşmesi doğrulandı. Sekiz örneğin üçü Baseten, dördü Fireworks, biri CoreWeave üzerinden yazıldı.

| Son doğrulama | Sonuç |
|---|---:|
| Sekiz refine, medyan / en uzun | 5,346 / 12,686 sn |
| Sekiz yazım isteği, medyan / en uzun | 2,782 / 9,516 sn |
| Sekiz yazım isteğinin toplam bedeli | $0,00285966 |
| Aday imajındaki ayrı analiz | 12,792 sn |
| Dağıtım sonrası public HTTPS soğuk refine | 10,898 sn |
| Aynı public refine içindeki yazım | 1,467 sn / $0,000285813 |
| Dağıtım sonrası public HTTPS ayrı analiz | 21,024 sn |

Bedeller **yalnız yazım isteğidir**; Jev ve analiz maliyeti bu yanıtlarda verilmez. İki tur farklı sağlayıcı kapasitesiyle çalıştı; tablodan kontrollü hız üstünlüğü veya sürekli gecikme garantisi çıkarılmaz. Güçlendirme ve analiz ayrı işlemlerdir. Bazı yaratıcı adaylar desteksiz ayrıntı nedeniyle elendi; bazı seçilenler kaynakla aynı veya çok yalın kaldı. Son public davet de yalın bir soru olarak seçildi. Bir adayda ikinci retorik soru editörden geçti. Dolayısıyla bu geçiş **her örnekte daha ikna edici metin** kanıtı değildir. Analiz anlatısındaki ilgi/tepki yorumları da model varsayımıdır; bireysel fMRI veya davranış ölçümü sayılmaz.

Canlı kaynak **`8210fae1d9c09bd8db7e29dcd4425e11a6e9e2b6`**, imaj **`sha256:601ae4665476244c93abbebd699b34be5ce80d747b7c4e5bfc51876a50f2796d`**. `/app/tribe_service` içindeki 28 izlenen dosyanın SHA-256 değerleri bu kaynakla eşleşti. Aynı CUDA bağımlılık imajında **174 Python testi** geçti; hafif yerelde 170 geçti/4 bağımlılık testi atlandı. Mobilde **136 Flutter testi** ve analiz geçti; CI native iOS güvenlik/owner akışı, imzalama ve IPA üretimini tamamladı.

Mobil kaynak **`40ce6af826c6ab5202c0a54df387e4e6324cafcb`**, [Codemagic derlemesi `6ac366ce70f9b1c8e1a9db28`](https://codemagic.io/app/6a6e4304766592d1996c4c2b/build/6ac366ce70f9b1c8e1a9db28), **1.0.0 (1791191321)** olarak Apple tarafından işlendi ve yalnız mevcut **Taskfulight Owner** grubuna eklendi. App Store yayını yapılmadı; bu turda fiziksel iPhone etkileşimi denenmedi.

Yalnız `tribe` servisi yenilendi; mevcut 4 CPU / 12 GiB RAM / 8 GiB CUDA sınırları korundu. Dağıtım anındaki diğer **173 konteyner** değişmedi. Çalışma sırasında ayrı bir `systemtest2` çalışması yeni konteynerler başlattı: son adayın ham tam-snapshot kontrolü 168→172 sayımı nedeniyle `ok=false` verdi; bütün mevcut 168 satır aynıydı ve gerçek API kabulü `ok=true` idi. Ham sonuç korunup ayrı `concurrent-service-verification.json` açıklaması eklendi. Public kontrol sonrasında aynı dış test çalışmasının worker sağlık durumu değişti; sonraki salt okunur gözlemde yeniden oluşturulmuş ve sağlıklıydı. O servise müdahale edilmedi. Başlangıçtaki **165 kalıcı servisin** kimliği, başlangıç zamanı, restart ve sağlık durumu aynı kaldı.

Son sağlık kontrolü başarılı, aktif GPU işi **0**, hem TRIBE hem text encoder boşaltılmıştır; logout başarılıdır ve geçici aday konteyner kalmamıştır. Unload sonrası Docker working-set **2,033 GiB / 12 GiB** idi; bu süreç RSS veya bütün CPU belleğinin boşaldığı iddiası değildir.

Sunucu kökü `/home/dkmserver/Desktop/Machinity/aytug/pitchserver/` altında son aday kanıtı `audit-glm-candidate-20261005-122351-20ad4a03/`; canlı kaynak/HTTPS/servis kayıtları `audit-glm-final-20261005-8210fae/` dizinindedir. Hedefli geri dönüş **`rollback-glm-final-20261005-8210fae-20261005T092748Z/rollback.sh`** ile yapılır; önceki imajı ve korumalı runtime ayarlarını geri getirip yalnız `tribe` servisini yeniler. Sentetik metinlerin tam yanıtları özel sunucu kayıtlarında tutulur; token veya API anahtarı bu dokümana eklenmez.

## Uzun metin düzeltmesi ve Haiku karşılaştırması — 8 Ekim 2026

Kaynak `b245723d9dc0a3b7a695766aecd3fb44e53bf183`, uzun girdileri kısa davet kalıbına veya özete çeviren çelişkili yazım talimatlarını kaldırır. Bütün adaylar kaynak kelime sayısının yaklaşık %80–120 aralığını hedefler; kısa metinlerde en az beş kelimelik tolerans vardır. Anlam, destekleyici ayrıntılar, sayılar, koşullar ve her paragrafın kapsamı korunurken ifade ve düzen yeniden yazılır. Üç aday farklı açılışlar ve tanınabilir stratejiler kullanır. Aşırı kısa/uzun adaylar seçilemez; uygun iyileştirme yoksa kaynak korunur. Metinler sınırı tutturmak için kesilmez veya doldurulmaz.

Tek yazım çağrısının completion bütçesi kaynak karakter sayısıyla ölçeklenir: `min(65536, max(1536, ceil(chars * 3.6) + 768))`. Jev planı/editörü, `z-ai/glm-5.3-flash`, canonical `meta-llama/Llama-3.2-3B`, orijinal dahil dört gerçek TRIBE ölçümü ve mevcut seçim formülü korunmuştur. Yeniden üretim döngüsü eklenmemiştir. Mobil ve web çağıranların tam metni taşıdığı doğrulandı; mobil kod veya yeni TestFlight derlemesi gerekmez.

Satış geri bildirimi yalnız iş bağlamına uygulandı: verilen müşteri ihtiyacını asgari gereksinim olarak ortaya koy, sunulan çözümün bütün ürün grubunu kapsayıp kapsamadığını ve verilmiş kapsam açığını göster. Rakip adı, rakibin verilmemiş eksikleri, ölçü birimi, indirim veya garanti uydurulmaz. Görüşmedeki 1250/1000 değerleri ürün kurallarına sabitlenmez.

İlk uzunluk düzeltmesinde GLM'nin kaynağı üç kez aynen döndürdüğü hata ham yanıtla yakalandı. `4f55ffa` canlı HTTPS kontrolünü geçmedi ve önceki imaja hedefli geri dönüş yapıldı. Sonraki talimat anlamı korumak ile cümleleri kopyalamayı ayırır, üç farklı açılış/düzen ister ve mevcut dayanak/fikir alanlarının sınırlarını açıkça söyler. Tekillik ve olgusal doğrulama gevşetilmedi. Son CUDA imajında **192 Python testi** geçti; yerelde **188 geçti / 4 bağımlılık testi atlandı**. İlk değişiklikte web lint, TypeScript ve **49 web testi** de geçti; sonraki talimat düzeltmesi web kodunu değiştirmedi.

Yeni imajın kayıtlı gerçek Jev planıyla üç bağımsız yazım kontrolünde dokuz adayın tamamı farklıydı; hiçbiri kaynakla aynı değildi. 251 kelimelik girdiden **219–255 kelime** üretildi. Ayrı bir yeni Jev planıyla üç aday **250/256/251 kelime** oldu. Bunlar yeni GPU ölçümü içermeyen yazım kontrolleridir. Sonraki plan çağrısı `529`, bir tam akışın editör çağrısı da `503` verdi. Başka bir tam akış yazımdan sonra `502` verdi ancak ham yazar yanıtı kaydedilmediği için kesin nedeni belirlenemedi; sonraki kabul, özel tanılama kaydıyla çalıştırıldı. Bütün başarısız adaylar kaldırılıp önceki sağlıklı imaj geri getirildi; her iki son başarısız aday turunda diğer **184 konteyner** değişmedi.

### Sınırlı yazar A/B deneyi

[Haiku 5.5](https://www.anthropic.com/claude-haiku-5-5) için kısa bağlam liste fiyatı bir milyon girdi/çıktı token başına $0,10/$0,50; seçili hızlı [GLM sağlayıcılarında](https://openrouter.ai/z-ai/glm-5.3-flash) $0,15/$0,50 idi. Liste fiyatı bütün isteğin maliyetini belirlemez. Aynı iki kaynak ve kayıtlı gerçek Jev planlarıyla yalnız yazım karşılaştırıldı; yeni metinler bu A/B sırasında TRIBE ile yeniden ölçülmedi. Karşılaştırma son kopyalama karşıtı talimat düzeltmesinden öncedir.

| Yazıcı / düşünme ayarı | Kısa davet: süre / bedel | 251 kelimelik iş metni: süre / bedel |
|---|---:|---:|
| GLM-5.3-Flash / low | 2,768 sn / $0,00033804 | 18,018 sn / $0,00140823 |
| Haiku 5.5 / low | 10,328 sn / $0,00098485 | 23,393 sn / $0,00371745 |
| Haiku 5.5 / düşünme kapalı | 3,594 sn / $0,00045560 | 10,831 sn / $0,00195416 |

Haiku-low kısa örnekte bütün 1.536 completion tokenını düşünmeye harcadı ve JSON üretmedi. Düşünme kapalıyken uzun örneği hızlandı, ancak iki taslak aynıydı ve mevcut tekillik kontrolünden geçmedi; kısa örneğin bir adayında verilmemiş `bu akşam` zamanı vardı. Kopyalama ilk GLM talimatında da görüldüğünden bu hata Haiku'ya özgü sayılmaz. Haiku-off bu iki istekte daha çok token kullandı ve yaklaşık %35–39 daha pahalıya geldi. Küçük örneklem genel kalite sıralaması veya insanlarda ikna artışı kanıtı değildir; model geçişini haklı çıkaran bir avantaj görülmediği için GLM korunur. [OpenRouter reasoning ayarlarında](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) `exclude` düşünmeyi kapatmaz; Haiku'nun hızlı denemesinde ayrıca `enabled:false` kullanıldı ve desteklemediği temperature gönderilmedi.

A/B bedelleri yalnız yazıcı içindir; Jev dahil değildir. Ham sentetik metinler ve ölçümler sunucunun özel `releases/length-20261008/` ve `releases/length-20261008-v2/` dizinlerinde tutulur. API anahtarı veya auth tokenı rapora eklenmez.

### Tam akış ve son düzeltme

Kopyalama karşıtı ilk imajın üç vakalı gerçek ASGI/auth kabulü, gerçek Jev ve her vakada orijinal dahil dört canonical TRIBE ölçümüyle tamamlandı. 251 kelimelik iş metninde 232 kelimelik c1 seçildi; toplam 61,285 sn / yazıcı 13,631 sn sürdü. 439 kelimelik kişisel metnin adayları 453/455/446 kelimeydi; deneysel TRIBE katkısı seçim puanını düşürdüğünden kaynak korundu (toplam 129,395 sn / yazıcı 29,143 sn). Bu, yeni metnin daha uzun olmasıyla otomatik kazanmadığını gösterir. Kısa davette üç aday da emir, desteklenmeyen vaat veya geçmiş iddiası nedeniyle elendi; kaynak döndü. API kabulü dil kalitesi garantisi sayılmadı.

Son altı satırlık talimat düzeltmesi bu kısa örnek için önceliği açıklığa kavuşturur: kaynakta bulunan emir, baskı, abartı ve alıcının eğleneceğine dair desteksiz vaatler korunacak ayrıntılar değildir. Gerçek amaç, olgular ve gönderenin hevesi korunur; heves gönderenin kendi bakışından anlatılır. Jev/eski yol için tek çağrı ve bu önceliği doğrulayan regresyon testi eklendi. Model, eşikler, eleştirmen ve orijinale dönüş politikası değişmedi.

Üç vakalı kabulün kayıtları `audit-length-instrumented-20261008-160904-371c3843/` altındadır. `glm-acceptance.json` ve tanılama başarılıdır. Ham bakım sonucu, eşzamanlı `systemtest2/systemtest3` konteynerlerinin beş satırı değiştiği için `ok=false` kaldı; 179 mevcut satır değişmedi. Salt okunur sonraki kontrolde değişen beş servis sağlıklı, `OOMKilled=false` ve restart sayıları 0 idi. Ham kanıt korunarak `concurrent-service-verification.json` eklendi; operatör yalnız PitchServer üzerinde stop/run/remove/start yaptı.

### Canlı son kabul ve geri dönüş

Son kaynak **`b245723d9dc0a3b7a695766aecd3fb44e53bf183`**, imaj **`sha256:b393b92f4ced9b3740400c9bf97aaf1f2753eab3f8e0cb85ba9043213c817bd0`**. Altı satırlık son öncelik düzeltmesinden sonra tam Python paketi aynı CUDA imajında yeniden geçti (**192 test**). Değişiklikten doğrudan etkilenen uzun iş metni ve kısa davet için iki gerçek ASGI/auth kabulü çalıştırıldı; önceki tamamlanmış 439 kelimelik test gereksiz yere tekrarlanmadı. Final aday kontrolü ve diğer servislerin tam snapshot karşılaştırması başarılıdır.

| Son doğrulama | Kaynak → seçilen kelime | Toplam süre | Tek yazım süresi |
|---|---:|---:|---:|
| İzole son aday: iş metni | 251 → 224 | 52,881 sn | 18,057 sn |
| İzole son aday: kısa davet | 9 → 8 | 5,778 sn | 3,641 sn |
| Dağıtım sonrası public HTTPS: iş metni | 251 → 242 | 51,623 sn | 15,798 sn |
| Dağıtım sonrası public HTTPS: kısa davet | 9 → 12 | 13,205 sn | 10,823 sn |

Bu dört isteğin her birinde tek GLM yazımı, gerçek Jev planı/editörü ve **dört gerçek 20.484 voxel TRIBE ölçümü** doğrulandı. Eski mobil istemcinin Gemini model kimliği de gerçek GLM yazımına yönlendi. Son public kısa mesaj: “Çilekeş konserine birlikte gidelim mi? O kadar heyecanlıyım ki seninle izlemek isterim.” Bu örnekte emir ve alıcının eğleneceğine dair vaat kalktı; her metinde aynı kalite sonucu garanti edilmez. Public iki yazımın toplam maliyeti **$0,00175091**, Jev hariçtir.

Yalnız `tribe` yenilendi. Dağıtımda ve public testler sonrasındaki son karşılaştırmada diğer **184 konteynerin** kimliği, başlangıç zamanı, restart ve sağlık durumu aynı kaldı. `/app/tribe_service` altındaki **28 izlenen dosyanın SHA-256 değeri** kaynak commit'iyle eşleşti. 4 CPU / 12 GiB RAM / 12 GiB toplam RAM+swap / 256 PID / 8 GiB PyTorch CUDA allocator sınırları korundu. Sağlık kontrolü başarılı, aktif GPU işi **0**, TRIBE ve text encoder boşaltılmış, logout `200`; geçici aday konteyner kalmadı. Docker working-set unload sonrası **1,658 GiB / 12 GiB** idi; bu süreç RSS veya CPU belleğinin tamamının serbest kalması demek değildir.

Kanıtlar sunucu kökü `/home/dkmserver/Desktop/Machinity/aytug/pitchserver/` altında `audit-length-final-candidate-20261008-162045-1d962b5d/` ve `audit-length-v3-final-20261008-b245723/` dizinlerindedir. İkinci dizin `deployment.json`, `public-length.json`, `source-verification.json`, `final-verification.json` içerir. Test edilen kaynak ve helper'lar `releases/length-20261008-v3/` altındadır. Önceki imaja ve korumalı runtime ayarlarına hedefli geri dönüş **`rollback-length-v3-final-20261008-b245723-20261008T132257Z/rollback.sh`** ile yalnız `tribe` üzerinde yapılır.

30.000 karakter sınırına yakın metinler ve fiziksel iPhone bu turda denenmedi. 439 kelimelik önceki tam akış 129 saniye sürdü; uzun metinlerde dört TRIBE hesabı önemli süre tutar, yalnız yazıcı değişimi bunu kaldırmaz. Dış Jev servisinde gözlenen geçici `503/529` hataları uygulama içi tekrar döngüsüyle gizlenmez. Kaynak koruma ile üslup/olgu değerlendirmesi semantik model kararlarıdır; kaynak benzeri uzunluk tek başına bütün ayrıntıların veya insanlarda ikna başarısının garantisi değildir.
