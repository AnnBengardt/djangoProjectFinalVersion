   function googleTranslateElementInit() {
            new google.translate.TranslateElement({ pageLanguage: 'ru', includedLanguages: 'en,es,fr,ru,ja', layout: google.translate.TranslateElement.InlineLayout.HORIZONTAL }, 'google_translate_element');
        }

        function triggerHtmlEvent(element, eventName) {
            var event;
            if (document.createEvent) {
                event = document.createEvent('HTMLEvents');
                event.initEvent(eventName, true, true);
                element.dispatchEvent(event);
            } else {
                event = document.createEventObject();
                event.eventType = eventName;
                element.fireEvent('on' + event.eventType, event);
            }
        }

        function updateLanguage(value) {
                var selectIndex = 0;
                var a = document.querySelector("#google_translate_element select");
                switch (value) {
                    case "en":
                        selectIndex = 0;
                        break;
                    case "es":
                        selectIndex = 1;
                        break;
                    case "fr":
                        selectIndex = 2;
                        break;

                    case "ru":
                        selectIndex = 3;
                        break;
                    case "ja":
                        selectIndex = 4;
                        break;

                }
                a.selectedIndex = selectIndex;
                a.dispatchEvent(new Event('change'));
            }